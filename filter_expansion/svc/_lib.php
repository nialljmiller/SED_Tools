<?php
// Shared code for the NJM SVO-compatible Filter Profile Service (fps.php, getdata.php).
// Interface modelled on the SVO Filter Profile Service (Rodrigo et al. 2012, 2020;
// Rodrigo & Solano 2024).  Data: catalog.json, built by build_catalog.py.
if (basename($_SERVER['SCRIPT_FILENAME'] ?? '') === '_lib.php') { http_response_code(404); exit; }

ini_set('serialize_precision', '-1');

const SERVICE_URL = 'https://nillmill.ddns.net/sed_tools/svo_fps/fps.php';
const GETDATA_URL = 'https://nillmill.ddns.net/sed_tools/svo_fps/getdata.php';
const MAX_PARAM_LEN = 512;
const RANGE_PARAMS = ['WavelengthRef', 'WavelengthMean', 'WavelengthEff', 'WavelengthMin',
                      'WavelengthMax', 'WidthEff', 'FWHM'];
const MAGSYS = ['Vega' => 'ZeroPointVega', 'AB' => 'ZeroPointAB', 'ST' => 'ZeroPointST'];

$DESC = [
    'WavelengthRef' => 'Reference wavelength. Defined as the same than the pivot wavelength, unless the filter owner specified one (see SpecifiedParams).',
    'WavelengthMean' => 'Mean wavelength. Defined as integ[x*filter(x) dx]/integ[filter(x) dx]',
    'WavelengthEff' => 'Effective wavelength. Defined as integ[x*filter(x)*vega(x) dx]/integ[filter(x)*vega(x) dx]. Empty when the NJM Vega spectrum does not cover the filter.',
    'WavelengthMin' => 'Minimum filter wavelength. Defined as the first lambda value with a transmission at least 1% of maximum transmission',
    'WavelengthMax' => 'Maximum filter wavelength. Defined as the last lambda value with a transmission at least 1% of maximum transmission',
    'WidthEff' => 'Effective width. Defined as integ[filter(x) dx]/max(filter)',
    'WavelengthCen' => 'Central wavelength. Defined as the midpoint of the outermost half-maximum points',
    'WavelengthPivot' => 'Pivot wavelength. Defined as sqrt{integ[W(x) dx]/integ[W(x) dx/x^2]}, W = filter for energy counters, x*filter for photon counters',
    'WavelengthPeak' => 'Peak wavelength. Defined as the lambda value with larger transmission',
    'WavelengthPhot' => 'Photon distribution based effective wavelength. Defined as integ[x^2*filter(x)*vega(x) dx]/integ[x*filter(x)*vega(x) dx]',
    'FWHM' => 'Full width at half maximum. Defined as the difference between the outermost wavelengths for which filter transmission is half maximum',
    'Facility' => 'Observational facility',
    'Instrument' => 'Instrument',
    'PhotSystem' => 'Photometric system',
    'DetectorType' => 'Detector type. 0: energy counter, 1: photon counter',
    'SpecifiedParams' => 'NJM: PARAMs whose value is the one published by the filter owner (info.json specified) rather than computed from the curve',
];

// name => [ucd, utype, unit, datatype]; SVO FPS order (2022), then NJM extras.
$FIELD_DEFS = [
    'FilterProfileService' => ['meta.ref.ivorn', 'PhotometryFilter.fpsIdentifier', '', 'char'],
    'filterID' => ['meta.id', 'photdm:PhotometryFilter.identifier', '', 'char'],
    'WavelengthUnit' => ['meta.unit', 'PhotometryFilter.SpectralAxis.unit', '', 'char'],
    'WavelengthUCD' => ['meta.ucd', 'PhotometryFilter.SpectralAxis.UCD', '', 'char'],
    'PhotSystem' => ['', 'photdm:PhotometricSystem.description', '', 'char'],
    'DetectorType' => ['', 'photdm:PhotometricSystem.detectorType', '', 'char'],
    'Band' => ['', 'photdm:PhotometryFilter.bandName', '', 'char'],
    'Instrument' => ['instr', '', '', 'char'],
    'Facility' => ['instr.obsty', '', '', 'char'],
    'ProfileReference' => ['', '', '', 'char'],
    'CalibrationReference' => ['', '', '', 'char'],
    'Description' => ['meta.note', 'photdm:PhotometryFilter.description', '', 'char'],
    'Comments' => ['meta.note', '', '', 'char'],
    'WavelengthRef' => ['em.wl', '', 'Angstrom', 'double'],
    'WavelengthMean' => ['em.wl', 'photdm:PhotometryFilter.SpectralAxis.Coverage.Location.Value', 'Angstrom', 'double'],
    'WavelengthEff' => ['em.wl.effective', '', 'Angstrom', 'double'],
    'WavelengthMin' => ['em.wl;stat.min', 'photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Start', 'Angstrom', 'double'],
    'WavelengthMax' => ['em.wl;stat.max', 'photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Stop', 'Angstrom', 'double'],
    'WidthEff' => ['instr.bandwidth', 'photdm:PhotometryFilter.SpectralAxis.Coverage.Bounds.Extent', 'Angstrom', 'double'],
    'WavelengthCen' => ['em.wl', '', 'Angstrom', 'double'],
    'WavelengthPivot' => ['em.wl', '', 'Angstrom', 'double'],
    'WavelengthPeak' => ['em.wl', '', 'Angstrom', 'double'],
    'WavelengthPhot' => ['em.wl', '', 'Angstrom', 'double'],
    'FWHM' => ['instr.bandwidth', '', 'Angstrom', 'double'],
    'Fsun' => ['phot.flux.density', '', 'erg/cm2/s/A', 'double'],
    'PhotCalID' => ['meta.id', 'photdm:PhotCal.identifier', '', 'char'],
    'MagSys' => ['meta.code', 'photdm:PhotCal.MagnitudeSystem.type', '', 'char'],
    'ZeroPoint' => ['phot.flux.density', 'photdm:PhotCal.ZeroPoint.Flux.value', 'Jy', 'double'],
    'ZeroPointUnit' => ['meta.unit', 'photdm:PhotCal.ZeroPoint.Flux.unit', '', 'char'],
    'Mag0' => ['phot.mag', 'photdm:PhotCal.ZeroPoint.referenceMagnitude.value', '', 'double'],
    'ZeroPointType' => ['meta.code', 'photdm:PhotCal.ZeroPoint.type', '', 'char'],
    'AsinhSoft' => ['obs.param', 'photdm:PhotCal.AsinhZeroPoint.softeningParameter', '', 'double'],
    'TrasmissionCurve' => ['DATA_LINK', 'phfdm:PhotometryFilter.filterTransmissionCurve', '', 'char'],
    'Components' => ['meta.note', '', '', 'char'],
    'ComponentsSource' => ['meta.note', '', '', 'char'],
    'DetectorTypeSource' => ['meta.note', '', '', 'char'],
    'NJMID' => ['meta.id', '', '', 'char'],
    'SpecifiedParams' => ['meta.note', '', '', 'char'],
];

$ACKS = [];   // catalog acknowledgements (from info.json), printed by head_xml()
$NOTES = [];  // extra INFO elements [name, value, text], printed by head_xml()

function x($s) {  // XML-escape, dropping characters XML 1.0 cannot carry
    $s = (string)$s;
    if (!mb_check_encoding($s, 'UTF-8')) $s = mb_convert_encoding($s, 'UTF-8', 'ISO-8859-1');
    $s = preg_replace('/[^\x{9}\x{A}\x{D}\x{20}-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}]/u', '', $s);
    return htmlspecialchars($s, ENT_XML1 | ENT_QUOTES, 'UTF-8');
}

function load_catalog() {
    $path = __DIR__ . '/catalog.json';
    $raw = @file_get_contents($path);
    $cat = $raw === false ? null : json_decode($raw, true);
    if (!is_array($cat) || !isset($cat['fields'], $cat['rows'])) return null;
    $cat['col'] = array_flip($cat['fields']);
    $cat['aliases'] = $cat['aliases'] ?? [];
    $cat['photcal'] = $cat['photcal'] ?? [];
    $GLOBALS['ACKS'] = is_array($cat['acknowledgements'] ?? null) ? $cat['acknowledgements'] : [];
    $cat['ids'] = [];
    $fi = $cat['col']['filterID'];
    foreach ($cat['rows'] as $i => $r) if (!isset($cat['ids'][$r[$fi]])) $cat['ids'][$r[$fi]] = $i;
    return $cat;
}

function rec($cat, $i) { return array_combine($cat['fields'], $cat['rows'][$i]); }

// Exact filterID, then an alias (NJM path ID Facility/Instrument.Band, info.json SVO
// filterID, or an SVO-form ID from svo_aliases.json), then a unique case-insensitive
// match on either.  Returns the row index or null.
function find_filter($cat, $id) {
    if (isset($cat['ids'][$id])) return $cat['ids'][$id];
    if (isset($cat['aliases'][$id])) return $cat['aliases'][$id];
    $lid = strtolower($id); $hit = [];
    foreach ($cat['ids'] as $k => $i) if (strtolower($k) === $lid) $hit[$i] = 1;
    foreach ($cat['aliases'] as $k => $i) if (strtolower($k) === $lid) $hit[$i] = 1;
    return count($hit) === 1 ? array_key_first($hit) : null;
}

function curve_url($fid) { return SERVICE_URL . '?ID=' . str_replace('%2F', '/', rawurlencode($fid)); }

// Default magnitude system of a filter: info.json MagSys, else Vega (as SVO lists most filters).
function default_magsys($rec) {
    $m = $rec['MagSys'] ?? '';
    return isset(MAGSYS[$m]) ? $m : 'Vega';
}

// Values for one filter in FIELD order; $magsys selects the zero point (Vega, AB, ST).
// info.json's PhotCalID and ZeroPointType describe its own MagSys only.
function out_values($rec, $magsys = null) {
    global $FIELD_DEFS;
    $magsys = $magsys ?? default_magsys($rec);
    $zp = $rec[MAGSYS[$magsys]];
    $own = ($rec['MagSys'] ?? '') === $magsys;
    $zpt = ($own && ($rec['ZeroPointType'] ?? '') !== '') ? $rec['ZeroPointType'] : 'Pogson';
    $v = [];
    foreach ($FIELD_DEFS as $name => $d) {
        switch ($name) {
            case 'PhotCalID': $v[$name] = ($own && ($rec['PhotCalID'] ?? '') !== '') ? $rec['PhotCalID'] : $rec['filterID'] . '/' . $magsys; break;
            case 'MagSys': $v[$name] = $magsys; break;
            case 'ZeroPoint': $v[$name] = $zp; break;
            case 'ZeroPointUnit': $v[$name] = 'Jy'; break;
            case 'Mag0': $v[$name] = 0; break;           // ZeroPoint is the flux of magnitude 0
            case 'ZeroPointType': $v[$name] = $zpt; break;
            case 'AsinhSoft': $v[$name] = strcasecmp($zpt, 'Pogson') === 0 ? 0 : null; break;  // 0 as SVO lists for Pogson; unknown otherwise
            case 'TrasmissionCurve': $v[$name] = curve_url($rec['filterID']); break;
            default: $v[$name] = $rec[$name] ?? null;
        }
    }
    return $v;
}

function fmt($v) {
    if ($v === null) return '';
    if (is_float($v)) return is_finite($v) ? json_encode($v) : '';   // shortest round-trip form
    return (string)$v;
}

function head_xml($status = 'OK', $msg = '') {
    header('Content-Type: text/xml; charset=UTF-8');
    header('Access-Control-Allow-Origin: *');
    $s = "<?xml version=\"1.0\"?>\n<VOTABLE version=\"1.1\" xsi:schemaLocation=\"http://www.ivoa.net/xml/VOTable/v1.1\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">\n";
    $s .= $msg === '' ? "  <INFO name=\"QUERY_STATUS\" value=\"" . x($status) . "\"/>\n"
                      : "  <INFO name=\"QUERY_STATUS\" value=\"" . x($status) . "\">" . x($msg) . "</INFO>\n";
    global $ACKS, $NOTES;
    foreach ($NOTES as [$n, $v, $t]) $s .= "  <INFO name=\"" . x($n) . "\" value=\"" . x($v) . "\">" . x($t) . "</INFO>\n";
    // SVO credit: info.json acknowledgements (text + references) via catalog.json;
    // the fixed citation is only a fallback for a catalog built without them.
    $s .= "  <INFO name=\"SERVICE\" value=\"NJM SVO-compatible Filter Profile Service; interface after the SVO Filter Profile Service"
        . ($ACKS ? '' : ' (Rodrigo et al. 2012, 2020; Rodrigo &amp; Solano 2024)')
        . ". See https://nillmill.ddns.net/sed_tools/svo_fps/README.html\"/>\n";
    foreach ($ACKS as $a) {   // citation = text + ". References: label (url); ..."
        $s .= "  <INFO name=\"ACKNOWLEDGEMENT\" value=\"" . x($a['for'] ?? '') . "\">"
            . x($a['citation'] ?? $a['text'] ?? '') . "</INFO>\n";
    }
    return $s;
}

// Error or empty answer: a VOTable without TABLE, so astroquery raises its own
// "No filter found ..." IndexError, as it does for SVO.
function fail($msg, $status = 'ERROR', $code = 200) {
    http_response_code($code);
    echo head_xml($status, $msg), "  <RESOURCE type=\"results\"/>\n</VOTABLE>\n";
    exit;
}

function field_xml($name, $tag = 'FIELD', $value = null) {
    global $FIELD_DEFS, $DESC;
    [$ucd, $utype, $unit, $dt] = $FIELD_DEFS[$name];
    $s = "    <$tag name=\"" . x($name) . "\"";
    if ($tag === 'PARAM') $s .= ' value="' . x(fmt($value)) . '"';
    $s .= " ucd=\"" . x($ucd) . "\" utype=\"" . x($utype) . "\" unit=\"" . x($unit) . "\" datatype=\"$dt\"";
    if ($dt === 'char') $s .= ' arraysize="*"';
    if ($tag === 'PARAM' && isset($DESC[$name])) return $s . ">\n       <DESCRIPTION>" . x($DESC[$name]) . "</DESCRIPTION>\n    </PARAM>\n";
    return $s . "/>\n";
}

// Read an NJM curve: [[wavelength_string, transmission_string], ...] or null.
function read_curve($cat, $i) {
    $root = realpath(__DIR__ . '/../filters');
    $path = realpath($root . '/' . $cat['files'][$i]);   // paths come from the catalog, not the user
    if ($root === false || $path === false || strpos($path, $root . DIRECTORY_SEPARATOR) !== 0) return null;
    $fh = @fopen($path, 'r');
    if (!$fh) return null;
    $out = [];
    while (($line = fgets($fh)) !== false) {
        $p = preg_split('/[\s,]+/', trim($line), 3);
        if (count($p) < 2 || !is_numeric($p[0]) || !is_numeric($p[1])) continue;  // header row
        $out[] = [$p[0], $p[1]];
    }
    fclose($fh);
    return $out;
}

// Single-valued string parameter, or null.  Arrays and over-long values are rejected.
function param($name) {
    static $req = null;
    if ($req === null) {
        $req = [];
        foreach (array_merge($_POST, $_GET) as $k => $v) $req[strtolower((string)$k)] = $v;
    }
    $k = strtolower($name);
    if (!array_key_exists($k, $req)) return null;
    $v = $req[$k];
    if (!is_string($v)) fail("Invalid value for $name");
    if (strlen($v) > MAX_PARAM_LEN) fail("Value too long for $name");
    return trim($v);
}
