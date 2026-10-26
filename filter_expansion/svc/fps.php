<?php
// NJM SVO-compatible Filter Profile Service: fps.php
// Same query interface as SVO FPS fps.php (Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024):
//   ?ID=Facility/Instrument.Band          filter PARAMs + Wavelength/Transmission table
//     (FilterID, NJM path ID, or an SVO-form ID mapped by info.json / svo_aliases.json)
//   ?PhotCalID=Facility/Instrument.Band/Vega|AB|ST
//   ?Facility=..[&Instrument=..][&PhotSystem=..]  filter list
//   ?<X>_min=..&<X>_max=..  or  ?<X>=min/max, X in WavelengthRef, WavelengthMean,
//     WavelengthEff, WavelengthMin, WavelengthMax, WidthEff, FWHM (Angstrom)
//   no parameters, or FORMAT=metadata     service description (INPUT params)
require __DIR__ . '/_lib.php';

$cat = load_catalog();
if ($cat === null) fail('Service catalog unavailable', 'ERROR', 503);
header('Cache-Control: public, max-age=3600');

function single($cat, $i, $magsys) {
    global $FIELD_DEFS;
    $data = read_curve($cat, $i);
    if (!$data) fail('Transmission curve unavailable');
    $v = out_values(rec($cat, $i), $magsys);
    $s = head_xml() . "  <RESOURCE type=\"results\">\n    <TABLE utype=\"photdm:PhotometryFilter.transmissionCurve.spectrum\">\n";
    foreach ($FIELD_DEFS as $name => $d) {
        if ($name === 'TrasmissionCurve') continue;
        $s .= field_xml($name, 'PARAM', $v[$name]);
    }
    $s .= "      <FIELD name=\"Wavelength\" utype=\"spec:Data.SpectralAxis.Value\" ucd=\"em.wl\" unit=\"Angstrom\" datatype=\"double\"/>\n";
    $s .= "      <FIELD name=\"Transmission\" utype=\"spec:Data.FluxAxis.Value\" ucd=\"phys.transmission\" unit=\"\" datatype=\"double\"/>\n";
    $s .= "      <DATA>\n        <TABLEDATA>\n";
    echo $s;
    foreach ($data as [$w, $t]) echo "          <TR><TD>", x($w), "</TD><TD>", x($t), "</TD></TR>\n";
    echo "        </TABLEDATA>\n      </DATA>\n    </TABLE>\n  </RESOURCE>\n</VOTABLE>\n";
    exit;
}

function number_or_fail($name, $s) {
    if ($s === '') return null;
    if (!is_numeric($s) || !is_finite((float)$s)) fail("Invalid numeric value for $name");
    return (float)$s;
}

function metadata($cat) {
    global $FIELD_DEFS, $DESC;
    $s = head_xml() . "  <RESOURCE type=\"meta\">\n  <DESCRIPTION>Filter information service (NJM, SVO-compatible)</DESCRIPTION>\n";
    foreach (RANGE_PARAMS as $p) {
        [$lo, $hi] = $cat['ranges'][$p] ?? [null, null];
        [$ucd] = $FIELD_DEFS[$p];
        foreach (['min', 'max'] as $sfx) {
            $s .= "    <PARAM name=\"INPUT:{$p}_$sfx\" ucd=\"" . x($ucd) . "\" utype=\"\" unit=\"Angstrom\" value=\"\" datatype=\"double\" >\n"
                . "      <DESCRIPTION>" . x($DESC[$p]) . "</DESCRIPTION>\n      <VALUES>\n"
                . "        <MIN value=\"" . x(fmt($lo)) . "\"/>\n        <MAX value=\"" . x(fmt($hi)) . "\"/>\n      </VALUES>\n    </PARAM>\n";
        }
    }
    foreach (['Instrument' => 'instruments', 'Facility' => 'facilities', 'PhotSystem' => null] as $p => $key) {
        [$ucd] = $FIELD_DEFS[$p];
        $s .= "    <PARAM name=\"INPUT:$p\" ucd=\"" . x($ucd) . "\" utype=\"\" unit=\"\" value=\"\" datatype=\"char\" arraysize=\"*\">\n"
            . "      <DESCRIPTION>" . x($DESC[$p]) . "</DESCRIPTION>\n      <VALUES>\n";
        foreach ($key ? $cat[$key] : [] as $o) $s .= "        <OPTION value=\"" . x($o) . "\"/>\n";
        $s .= "      </VALUES>\n    </PARAM>\n";
    }
    $s .= "    <TABLE>\n";
    foreach ($FIELD_DEFS as $name => $d) $s .= field_xml($name);
    echo $s, "    </TABLE>\n  </RESOURCE>\n</VOTABLE>\n";
    exit;
}

// ---- dispatch ----
$format = param('FORMAT');
if ($format !== null && strtolower($format) === 'metadata') metadata($cat);

$id = param('ID');
if ($id !== null && $id !== '') {
    $i = find_filter($cat, $id);
    if ($i === null) fail('Filter ID not found: ' . $id);
    $fid = $cat['rows'][$i][$cat['col']['filterID']];
    if ($fid !== $id) $NOTES[] = ['ALIAS', $id, "Requested ID resolved to NJM filter $fid ("
        . $cat['rows'][$i][$cat['col']['NJMID']] . ') through the NJM alias table'];
    single($cat, $i, default_magsys(rec($cat, $i)));
}
$pc = param('PhotCalID');
if ($pc !== null && $pc !== '') {
    if (isset($cat['photcal'][$pc])) {   // info.json PhotCalID (may differ from filterID/MagSys)
        $i = $cat['photcal'][$pc];
        single($cat, $i, default_magsys(rec($cat, $i)));
    }
    $cut = strrpos($pc, '/');
    $ms = $cut === false ? '' : substr($pc, $cut + 1);
    $key = null;
    foreach (array_keys(MAGSYS) as $k) if (strcasecmp($k, $ms) === 0) $key = $k;
    $i = $key === null ? null : find_filter($cat, substr($pc, 0, $cut));
    if ($i === null) fail('PhotCalID not found: ' . $pc);
    single($cat, $i, $key);
}

$col = $cat['col'];
$conds = [];   // [column, min|null, max|null]
foreach (RANGE_PARAMS as $p) {
    $lo = param($p . '_min'); $hi = param($p . '_max'); $both = param($p);
    if ($both !== null && $both !== '') {
        $parts = explode('/', $both);
        if (count($parts) !== 2) fail("Invalid range for $p (use min/max)");
        [$lo, $hi] = $parts;
    }
    $lo = $lo === null ? null : number_or_fail($p, trim($lo));
    $hi = $hi === null ? null : number_or_fail($p, trim($hi));
    if ($lo !== null || $hi !== null) $conds[] = [$col[$p], $lo, $hi];
}
$text = [];
foreach (['Facility', 'Instrument', 'PhotSystem'] as $p) {
    $v = param($p);
    if ($v !== null && $v !== '') $text[$p] = strtolower($v);
}
if (!$conds && !$text) metadata($cat);

$njm = $col['NJMID'];
$hits = [];
foreach ($cat['rows'] as $i => $r) {
    foreach ($conds as [$c, $lo, $hi]) {
        $v = $r[$c];
        if ($v === null || ($lo !== null && $v < $lo) || ($hi !== null && $v > $hi)) continue 2;
    }
    if ($text) {
        // NJM directory names are accepted as well as SVO names (e.g. ADEOS POLDER vs POLDER-2).
        [$nf, $rest] = explode('/', $r[$njm], 2) + ['', ''];
        $ni = explode('.', $rest, 2)[0];
        if (isset($text['Facility']) && strtolower($r[$col['Facility']]) !== $text['Facility'] && strtolower($nf) !== $text['Facility']) continue;
        if (isset($text['Instrument']) && strtolower($r[$col['Instrument']]) !== $text['Instrument'] && strtolower($ni) !== $text['Instrument']) continue;
        if (isset($text['PhotSystem']) && strtolower($r[$col['PhotSystem']]) !== $text['PhotSystem']) continue;
    }
    $hits[] = $i;
}
if (!$hits) fail('No filters match the query', 'OK');

$s = head_xml() . "  <RESOURCE type=\"results\">\n    <TABLE>\n";
foreach ($FIELD_DEFS as $name => $d) $s .= field_xml($name);
echo $s, "      <DATA>\n        <TABLEDATA>\n";
foreach ($hits as $i) {
    $s = "          <TR>";
    foreach (out_values(rec($cat, $i)) as $name => $v) {
        $s .= $name === 'TrasmissionCurve' ? '<TD><![CDATA[' . str_replace(']]>', '', $v) . ']]></TD>' : '<TD>' . x(fmt($v)) . '</TD>';
    }
    echo $s, "</TR>\n";
}
echo "        </TABLEDATA>\n      </DATA>\n    </TABLE>\n  </RESOURCE>\n</VOTABLE>\n";
