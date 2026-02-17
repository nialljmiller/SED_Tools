<?php
$file = __DIR__ . "/count.txt";

if (!file_exists($file)) {
    file_put_contents($file, "0");
}

$count = (int)file_get_contents($file);
$count++;
file_put_contents($file, (string)$count);

echo str_pad($count, 6, "0", STR_PAD_LEFT);
