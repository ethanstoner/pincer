# pogo-catcher

A PC tool that drives Android phones over ADB to automatically catch Pokemon in the game, running one worker per connected phone. Each worker screenshots the device, detects map/encounter state, and performs the tap/throw gestures needed to catch Pokemon without manual input.

Requires: USB debugging authorized on each phone, ADB (platform-tools) installed, and the Total Control app closed on the phone before starting a worker.
