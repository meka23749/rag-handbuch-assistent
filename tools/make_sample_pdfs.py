"""Create two small FICTIONAL machine manuals as test data.
 
The machines, codes and values below are invented for testing only.
"""
from pathlib import Path
 
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
 
OUT = Path(__file__).resolve().parent.parent / "data" / "pdfs"
 
MANUALS = {
    "kompressor_kx200.pdf": ("Betriebsanleitung Schraubenkompressor KX-200 (fiktives Testdokument)", [
        ("1 Sicherheit", "Vor allen Arbeiten den Kompressor am Hauptschalter ausschalten und gegen Wiedereinschalten sichern. "
         "Der Druckbehälter muss vollständig entlüftet sein, bevor Teile geöffnet werden. Arbeiten an elektrischen Teilen "
         "dürfen nur Elektrofachkräfte ausführen."),
        ("2 Technische Daten", "Maximaler Betriebsdruck: 10 bar. Liefermenge: 1,8 m³/min. Motorleistung: 11 kW. "
         "Zulässige Umgebungstemperatur: +5 °C bis +40 °C. Ölfüllmenge: 4,5 Liter."),
        ("PAGE", ""),
        ("3 Wartung", "Der Ansaugfilter ist alle 1000 Betriebsstunden zu prüfen und spätestens nach 2000 Betriebsstunden zu "
         "wechseln. Das Öl wird alle 4000 Betriebsstunden oder mindestens einmal jährlich gewechselt. Der Ölabscheider "
         "wird alle 6000 Betriebsstunden ersetzt. Die Riemenspannung ist monatlich zu kontrollieren."),
        ("4 Störungen und Fehlercodes", "E12: Übertemperatur Verdichter. Ursache meist verschmutzter Kühler oder zu wenig Öl. "
         "Kühler reinigen und Ölstand prüfen. E27: Drucksensor defekt oder Kabelbruch. Sensor und Leitung prüfen, "
         "gegebenenfalls Sensor tauschen. E42: Drehrichtung des Motors falsch. Zwei Phasen der Netzzuleitung durch eine "
         "Elektrofachkraft tauschen lassen. E55: Wartungsintervall überschritten. Wartung durchführen und Zähler zurücksetzen."),
    ]),
    "laser_lm50.pdf": ("Bedienungsanleitung Laserbeschriftungsanlage LM-50 (fiktives Testdokument)", [
        ("1 Allgemeines", "Die LM-50 beschriftet Metall- und Kunststoffteile mit einem Faserlaser der Klasse 4. "
         "Die Anlage darf nur mit geschlossener Schutzhaube betrieben werden."),
        ("2 Inbetriebnahme", "Netzstecker einstecken, Hauptschalter auf I stellen und warten, bis die Status-LED grün leuchtet. "
         "Danach das Beschriftungsprogramm am Bedienpanel auswählen und den Fokus mit der Fokuslehre einstellen."),
        ("PAGE", ""),
        ("3 Fehlermeldungen", "L01: Schutzhaube offen. Haube schließen, die Meldung verschwindet automatisch. "
         "L07: Laserquelle zu warm. Lüftungsgitter reinigen und Anlage 15 Minuten abkühlen lassen. "
         "L19: Keine Verbindung zur Steuerung. Netzwerkkabel und IP-Adresse der Steuerung prüfen."),
        ("4 Reinigung", "Die Schutzscheibe der Optik wöchentlich mit einem fusselfreien Tuch und Isopropanol reinigen. "
         "Absauganlage und Filter monatlich prüfen."),
    ]),
}
 
 
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    for name, (title, sections) in MANUALS.items():
        story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
        for heading, body in sections:
            if heading == "PAGE":
                story.append(PageBreak())
                continue
            story += [Paragraph(heading, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 8)]
        SimpleDocTemplate(str(OUT / name), pagesize=A4).build(story)
        print("created", OUT / name)
 
 
if __name__ == "__main__":
    main()