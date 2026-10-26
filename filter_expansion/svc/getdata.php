<?php
// NJM SVO-compatible Filter Profile Service: getdata.php
//   ?format=ascii&id=Facility/Instrument.Band   two columns: wavelength (Angstrom), transmission
//   ?format=votable&id=...                      same as fps.php?ID=...
// Interface after SVO FPS getdata.php (Rodrigo et al. 2012, 2020; Rodrigo & Solano 2024).
require __DIR__ . '/_lib.php';

function text_fail($code, $msg) {
    http_response_code($code);
    header('Content-Type: text/plain; charset=UTF-8');
    echo $msg, "\n";
    exit;
}

$id = $_GET['id'] ?? $_GET['ID'] ?? '';
$format = strtolower(is_string($_GET['format'] ?? null) ? $_GET['format'] : 'ascii');
if (!is_string($id) || $id === '' || strlen($id) > MAX_PARAM_LEN) text_fail(400, 'Missing or invalid id');
if ($format === 'votable') {
    header('Location: ' . curve_url($id), true, 302);
    exit;
}
if ($format !== 'ascii') text_fail(400, 'Unsupported format (use ascii or votable)');

$cat = load_catalog();
if ($cat === null) text_fail(503, 'Service catalog unavailable');
$i = find_filter($cat, $id);
if ($i === null) text_fail(404, 'Filter ID not found');
$data = read_curve($cat, $i);
if (!$data) text_fail(404, 'Transmission curve unavailable');

header('Content-Type: text/plain; charset=UTF-8');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: public, max-age=3600');
$out = '';
foreach ($data as [$w, $t]) $out .= $w . ' ' . $t . "\n";
echo $out;
