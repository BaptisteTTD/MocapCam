"""Ligne de commande MocapCam.

  python -m mocapcam init        STUDIO
  python -m mocapcam intrinseques STUDIO CAMERA VIDEO_DAMIER
  python -m mocapcam marqueurs   STUDIO DOSSIER_PRISE
  python -m mocapcam traiter     STUDIO DOSSIER_PRISE [--force] [--decalage poco=0.12]
"""
import argparse
import os
import sys

from . import intrinsics, pipeline, video

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mocapcam", description="Motion capture multi-téléphones")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="crée le dossier studio (studio.json)")
    p.add_argument("studio")

    p = sub.add_parser("studio", help="interface de pilotage des téléphones")
    p.add_argument("--studio", default=os.path.join(ROOT, "studio"))
    p.add_argument("--captures", default=os.path.join(ROOT, "captures"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--sans-fenetre", action="store_true", help="ne pas ouvrir l'interface")

    p = sub.add_parser("damier", help="(optionnel) crée l'image damier.png")
    p.add_argument("studio")

    p = sub.add_parser("intrinseques", help="(optionnel) calibration optique au damier")
    p.add_argument("studio")
    p.add_argument("camera")
    p.add_argument("video")

    for name, hlp in (("marqueurs", "vérifie la détection des boules et la position des caméras"),
                      ("traiter", "traite une prise complète -> fichier .mocap.json")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("studio")
        p.add_argument("prise")
        p.add_argument("--force", action="store_true", help="refaire tous les calculs")
        p.add_argument("--decalage", action="append", default=[],
                       help="décalage manuel en secondes, ex. poco=0.120")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "studio":
            from . import studio_server
            studio_server.run(a.studio, a.captures, a.port, open_ui=not a.sans_fenetre)
        elif a.cmd == "init":
            path = pipeline.init_studio(a.studio)
            print(f"Studio créé : {path}\nMesurez la position de vos boules et éditez ce fichier.")
        elif a.cmd == "damier":
            os.makedirs(a.studio, exist_ok=True)
            print(intrinsics.make_checkerboard(os.path.join(a.studio, "damier.png")))
        elif a.cmd == "intrinseques":
            cfg = pipeline.load_studio(a.studio)
            tmp = os.path.join(a.studio, "intrinseques", f"_norm_{a.camera}.mp4")
            print("Normalisation de la vidéo…")
            video.normalize(a.video, tmp, cfg["fps"], cfg["max_height"])
            print("Recherche du damier…")
            data = intrinsics.calibrate_from_video(tmp)
            data["camera"] = a.camera
            out = os.path.join(a.studio, "intrinseques", f"{a.camera}.json")
            intrinsics.save(out, data)
            print(f"Enregistré : {out}")
        else:
            manual = {}
            for d in a.decalage:
                k, v = d.split("=")
                manual[k.strip()] = float(v)
            pipeline.process_take(a.studio, a.prise, force=a.force, manual_delays=manual,
                                  only_calib=(a.cmd == "marqueurs"))
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\nERREUR : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
