"""MocapCam : motion capture sans marqueurs multi-caméras (traitement différé).

Pipeline :
  vidéos téléphones -> normalisation (CFR) -> synchro audio (clap)
  -> calibration auto par boules colorées au sol -> pose 2D corps + mains (RTMW)
  -> triangulation 3D -> nettoyage / lissage -> fichier .mocap.json
  -> add-on Blender (squelette Mixamo + application sur un perso Mixamo)
"""

__version__ = "0.1.0"
