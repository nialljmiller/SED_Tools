<?php
$file = __DIR__ . "/count.txt";

// One visit at a time: read and write the count under an exclusive lock,
// so two simultaneous visits cannot both read the old number.
$handle = fopen($file, "c+");
if ($handle === false) {
    exit;
}
flock($handle, LOCK_EX);
$count = (int)stream_get_contents($handle);
$count++;
ftruncate($handle, 0);
rewind($handle);
fwrite($handle, (string)$count);
fflush($handle);
flock($handle, LOCK_UN);
fclose($handle);

echo str_pad($count, 6, "0", STR_PAD_LEFT);
