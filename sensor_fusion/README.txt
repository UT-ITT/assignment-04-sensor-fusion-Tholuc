Moin Tutoren,

Der Filter funktioniert so (Zeile 254) pred_pos = alpha * (pred_pos + acc_delta) + (1 - alpha) * cam_pos
Wir nehmen also normal einen Teil von der Kamera und einen berechnen wir anhand der integrierten beschleunigung acc_delta = acc_vel * dt wie weit wir uns bewegen würden.


Hatte Alpha so das 0 camera weight is und 1 accelerator. wenn ich über 0.7 war haben sich die Rundungsfehler schnell aufadiert.
Bei .05 - 0.15 lief gut und die Prediction ist nah an der Camera und wirkt untertstützend.

Die Aufgabe war iwie mega schwierig und ich glaube es war teilweise mein equpiment weil ich für zuhause so ne super alte webcam hab und hab dann mit tesa den arucomarker an mein handy geklebt :D Iwie hatte mein Handy auch en weirden bias... War dann aber auch sehr überarbeitet die Woche und hab bei task 3 etwas mehr mit AI ausgeholfen. Finde man merkt es und bin nicht so zufrieden. Die Scorebedinungen sollten alle klappen aber wirklich polished ist es nicht. Hoffe die anderen Aufgabe gefallen mehr besonders das AR Game. 

War auch etwas verwirrt mit der Aufgabenstellung am Ende. Hätten die Sensoren unterschiedlich schnell im Code updaten sollen? weil iwie ist ja der Sinn von Complementary Filtern nen langsamen oder reliable sensor mit nem schnellen zu verbinden bzw unreliable/reliable ? Aber an sich updateted dippyd und die camera ja gleich schnell. Hatte schon überlegt das zu machen wenn die Kamera außerhalb des rectangles geht das die accelerator carried aber das hat auch nicht funktioniert, aber das hatte niels ja auch schon erklärt.. bisschen verwirrt.