<?php
/**
 * wl_cut.php — Server-side wavelength trimming for SED_Tools
 *
 * Streams a spectrum file with only wavelength rows within [wl_min, wl_max].
 * Headers/comments (#-prefixed lines) are always preserved.
 *
 * Usage:
 *   /sed_tools/wl_cut.php?model=bbody&file=spectrum_001.txt&wl_min=3000&wl_max=10000
 *
 * Place in: /media/data3/MESA/SED_Tools/data/wl_cut.php
 *           (same directory as index.json)
 */

// ── Validate parameters ──
$model = isset($_GET['model']) ? $_GET['model'] : null;
$file  = isset($_GET['file'])  ? $_GET['file']  : null;
$wl_min = isset($_GET['wl_min']) ? floatval($_GET['wl_min']) : null;
$wl_max = isset($_GET['wl_max']) ? floatval($_GET['wl_max']) : null;

if (!$model || !$file || $wl_min === null || $wl_max === null) {
    http_response_code(400);
    echo "Usage: ?model=NAME&file=FILENAME&wl_min=MIN&wl_max=MAX\n";
    exit;
}

// ── Security: prevent directory traversal ──
if (strpos($model, '..') !== false || strpos($file, '..') !== false ||
    strpos($model, '/') !== false  || strpos($file, '/') !== false  ||
    strpos($model, '\\') !== false || strpos($file, '\\') !== false) {
    http_response_code(403);
    echo "Invalid path\n";
    exit;
}

// Only allow .txt files
if (substr($file, -4) !== '.txt') {
    http_response_code(400);
    echo "Only .txt spectrum files supported\n";
    exit;
}

// ── Build path and check existence ──
$base = __DIR__ . '/stellar_models';
$path = "$base/$model/$file";

if (!is_file($path) || !is_readable($path)) {
    http_response_code(404);
    echo "File not found\n";
    exit;
}

// ── Stream trimmed content ──
header('Content-Type: text/plain; charset=utf-8');
header('Cache-Control: public, max-age=86400');
header('X-WL-Cut: ' . $wl_min . '-' . $wl_max);

$fh = fopen($path, 'r');
if (!$fh) {
    http_response_code(500);
    echo "Cannot open file\n";
    exit;
}

while (($line = fgets($fh)) !== false) {
    $trimmed = ltrim($line);

    // Always pass through headers, comments, and blank lines
    if ($trimmed === '' || $trimmed[0] === '#') {
        echo $line;
        continue;
    }

    // Parse wavelength from first column
    // Use sscanf for speed — only reads the first float
    $wl = null;
    if (sscanf($trimmed, '%f', $wl) === 1) {
        if ($wl >= $wl_min && $wl <= $wl_max) {
            echo $line;
        }
    } else {
        // Unparseable line — pass through (column headers, etc.)
        echo $line;
    }
}

fclose($fh);
