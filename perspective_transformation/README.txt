Moin Tutoren,

Ihr benutzt das Programm mit:

python image_extractor.py sample_image.jpg result.jpg --width 1000 --height 700

ESC Reset
Q   Abbruch
S   Speichern

Ich hatte erst en Bug der manchmal alles einfarbig gemacht hat weil ich das Rectangle blöd geordert habe. Das habe ich gefixed mit beliebiger Reihenfolge.
Zuerst hatte ich Linien einfach per Reihenfolge gezogen aber dann haben am Ende die Linien nicht mehr das Rechteck repräsentiert. Also wenn man so zickzack Punkte macht.
Das ist weil quasi per (x,y) Ordering sortiert habe. 
Deshalb passe ich die Linien beim Adden des vierten Punkts nochmal an um das Rectangle anzuzeigen was wir für die Perspektiventransformation nutzen.

Ich finde es hat eig alles so funktioniert und ausgesehen wie bei Christophs Präsentation.



I guess man hätte auch sich selbst schneidende Polygone als gültige Punkte nehmen. Ich dachte für den Scope dieser Aufgabe aber einfach an normale Rectangles weil die iwie Sinn ergeben.
Ich erlaube aber nicht konvexe Polygone wenn man mit den Punkten sich mühe gibt kein Rectangle zu machen. Ergebnisse sehen artsy aus 