<?php
/**
 * bulk_download.php — Stream a model directory as tar.gz
 *
 * Usage: /sed_tools/bulk_download.php?model=BT-Settl
 * Optional: ?model=BT-Settl&wl_min=3000&wl_max=10000  (filters via wl_cut logic)
 *
 * One HTTP request regardless of how many spectra are in the model.
 */

$model = isset($_GET['model']) ? $_GET['model'] : null;

if (!$model) {
    http_response_code(400);
    echo "Usage: ?model=NAME\n";
    exit;
}

// Security: no traversal, no slashes, no shenanigans
if (strpos($model, '..') !== false ||
    strpos($model, '/')  !== false ||
    strpos($model, '\\') !== false ||
    !preg_match('/^[\w\-]+$/', $model)) {
    http_response_code(403);
    echo "Invalid model name\n";
    exit;
}

$base_dir  = __DIR__ . '/stellar_models';
$model_dir = realpath($base_dir . '/' . $model);

// Path traversal guard — realpath must stay inside base_dir
if (!$model_dir || strpos($model_dir, realpath($base_dir)) !== 0) {
    http_response_code(404);
    echo "Model not found\n";
    exit;
}

$archive_name = $model . '.tar.gz';

header('Content-Type: application/gzip');
header('Content-Disposition: attachment; filename="' . $archive_name . '"');
header('X-Bulk-Download: ' . $model);
header('Cache-Control: public, max-age=86400');

// Stream tar directly to client — never touches disk
$cmd = 'tar -czf - -C ' . escapeshellarg($base_dir) . ' ' . escapeshellarg($model);
passthru($cmd);
