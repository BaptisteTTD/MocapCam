# MocapCam : motion capture avec deux téléphones, vers Blender

MocapCam transforme deux vidéos filmées au téléphone en une animation 3D, avec le corps
et les doigts : un **FBX au squelette Mixamo**, prêt pour Blender, Unity, Unreal…

Trois éléments :
* **l'app MocapCam** sur chaque téléphone : elle filme avec des réglages fixes (1080p 60 i/s,
  objectif ×1, stabilisation coupée, exposition bloquée) ;
* **MocapCam Studio** sur le PC : vous y voyez les caméras, vérifiez les boules, lancez et
  arrêtez l'enregistrement de tous les téléphones d'un clic, et les vidéos arrivent toutes seules ;
* **l'add-on Blender**, pour importer l'animation (FBX) et la transférer sur votre personnage.

```
iPhone + Poco ─► calage par le bip du PC ─► position + objectif des caméras grâce aux boules au sol
              ─► détection 2D corps + mains (IA RTMW) ─► synchro fine par le mouvement
              ─► affinage des caméras avec le corps
              ─► triangulation 3D ─► affinage global du mouvement (os rigides, fluidité, pieds posés)
              ─► prise.fbx (squelette Mixamo animé) ─► add-on Blender : retarget sur votre personnage
```

---

## 1. Installation (une fois)

1. Installez **Python 3.11 ou 3.12** depuis python.org, en cochant *Add python.exe to PATH*.
2. Double-cliquez sur **`installer.bat`**. Il crée l'environnement, installe les dépendances
   et prépare le dossier `studio\` avec `studio.json`.
3. Dans Blender : *Édition > Préférences > Modules complémentaires > ▾ > Installer depuis le disque*,
   choisissez `blender_addon\mocapcam_blender.py`, puis cochez **MocapCam**.

Au premier traitement, les modèles de détection (~200 Mo) se téléchargent automatiquement.

### Installer l'app sur les téléphones (une fois)

L'app est compilée **par GitHub**, gratuitement, sans Mac :

1. Lancez d'abord `installer.bat` : il met en place la recette de compilation
   (`.github\workflows\app-mobile.yml`). Créez ensuite un compte sur github.com.
   Avec **GitHub Desktop**, faites de `Documents\MocapCam`
   un dépôt (*File > Add local repository*, puis *create a repository*) et publiez-le
   (*Publish repository*). En dépôt **public**, la compilation iPhone est gratuite et illimitée.
   En privé, elle consomme le quota gratuit (environ 10 compilations iPhone par mois).
2. Sur la page du dépôt : onglet **Actions** > *App mobile MocapCam* > **Run workflow**.
   Comptez environ 15 min (Android) et 30 min (iPhone).
3. À la fin, en bas de la page du lancement, rubrique **Artifacts**, téléchargez :
   * `MocapCam-android` : il contient **MocapCam.apk** ;
   * `MocapCam-iphone` : il contient **MocapCam-non-signee.ipa**.

**Poco (Android)** : copiez `MocapCam.apk` sur le téléphone (câble USB ou Google Drive), ouvrez-le
et acceptez l'installation depuis une source inconnue.

**iPhone** (Apple ID gratuit, sans Mac) :
1. Sur le PC, installez **iTunes et iCloud depuis le site d'Apple**, pas les versions du
   Microsoft Store (désinstallez celles-ci si elles sont présentes), puis **Sideloadly** (sideloadly.io).
2. Branchez l'iPhone en USB, ouvrez iTunes une fois et touchez *Se fier à cet ordinateur* sur l'iPhone.
3. Dans Sideloadly : glissez `MocapCam-non-signee.ipa`, entrez votre Apple ID, cliquez **Start**.
4. Sur l'iPhone : *Réglages > Confidentialité et sécurité > Mode développeur* : activez-le, puis redémarrez.
   Ensuite, *Réglages > Général > VPN et gestion de l'appareil* : touchez votre Apple ID, puis **Faire confiance**.
5. **Tous les 7 jours**, l'app cesse de s'ouvrir, c'est la limite des Apple ID gratuits.
   Il suffit de refaire l'étape 3 avec le même fichier : rien n'est perdu.

## 2. Les boules au sol (une fois)

Il faut 5 demi-boules peintes de couleurs **vives et mates** (le brillant crée des reflets),
collées au sol avec de l'adhésif double face : 2 rouges, 1 verte, 1 bleue, 1 jaune.
Elles forment un **carré** dont la taille dépend de votre pièce (1,5 à 3 m de côté), plus une
5ᵉ boule au milieu d'un côté :

| Boule | Couleur | Position (côté du carré = L) |
|---|---|---|
| A | rouge | un coin : l'origine (0 ; 0) |
| B | verte | coin voisin de A : (L ; 0) |
| C | bleue | coin opposé à A : (L ; L) |
| D | jaune | l'autre coin voisin de A : (0 ; L) |
| E | rouge | **au milieu** du côté A-D : (0 ; L/2) |

* **Mesurez au mètre ruban**, d'un centre de boule à l'autre : les 4 côtés égaux, et les
  **2 diagonales égales** (= L × 1,41), sinon le carré est un losange. E doit être pile au milieu
  de A-D, sur la ligne. Une erreur de 2-3 cm est acceptable, 10 cm ne l'est plus.
* Indiquez le côté L dans MocapCam Studio : **Préparation > Carré des boules**, puis *Enregistrer*.
  Si la valeur est fausse, tout le mouvement est à la mauvaise échelle (le traitement l'affiche :
  « taille estimée de l'acteur »).
* Le sens n'a pas d'importance : le carré peut tourner dans un sens ou dans l'autre (B et C à
  gauche ou à droite de la ligne A-D), le logiciel s'en rend compte tout seul. Si deux boules de
  couleurs différentes sont échangées, il le détecte aussi et le signale.
* **X** va de A vers B, **Y** de A vers D, **Z** vers le haut. Le sol est à Z = 0.
* Chaque caméra doit voir **au moins 4 boules**, idéalement les 5. Si elle n'en voit que 4 dont
  3 alignées (A, E, D + une autre), ça marche, mais la mesure est moins sûre : Studio l'indique.
  Une boule cachée par l'acteur pendant la prise ne pose pas de problème.
* Le logiciel calcule la position des caméras **à chaque prise**, tout seul. Vous pouvez donc
  déplacer les trépieds d'une séance à l'autre sans rien recalibrer. Ne déplacez pas les boules
  entre deux prises (sinon remesurez).
* Il n'y a **aucune autre calibration à faire**, pas de damier. Les boules donnent une première
  estimation de la position et de l'objectif de chaque téléphone. Pendant le traitement, les
  mouvements de l'acteur, vus par les deux caméras, servent ensuite à affiner tout ça.
  Pour que cet affinage marche bien, **bougez dans toute la zone**, pas seulement au centre.

## 3. Réglages des téléphones

**Avec l'app MocapCam, il n'y a rien à régler** : elle impose elle-même 1080p 60 i/s, l'objectif ×1,
la stabilisation coupée et une image sans HDR. Studio affiche ce que chaque téléphone a
réellement accepté.

**Cadrage** : ce que montre Studio est exactement ce qui est filmé. Sur Android, l'app filme
en **4:3 (1440×1080)**, c'est-à-dire tout le capteur, comme l'aperçu : en 16:9, le téléphone
coupait le haut et le bas de l'image. L'iPhone filme en 16:9, identique à son aperçu. Si un
téléphone ne peut pas faire autrement, son écran assombrit les bords qui ne seront pas filmés.

Les réglages ci-dessous ne concernent que les vidéos filmées avec l'appareil photo normal (section 7) :

* **1080p, 60 i/s, en paysage**, avec l'objectif principal (×1, sans zoom)
* **Stabilisation désactivée**, car elle déforme l'image différemment à chaque instant
* **iPhone** : *Réglages > Appareil photo > Enregistrer une vidéo* : désactivez *Vidéo HDR*,
  et dans *Formats* choisissez *Le plus compatible*. Au tournage, faites un appui long sur
  l'écran pour verrouiller l'exposition et la mise au point (« Verrouillage AE/AF »).
* **Poco X3** : désactivez la stabilisation (EIS). Utilisez le mode vidéo Pro s'il est disponible
  pour bloquer l'exposition et la mise au point.
* Le LiDAR de l'iPhone n'est pas utilisé : il n'est pas enregistré dans une vidéo normale.

## 4. (Optionnel) Calibration au damier

Elle n'est **pas nécessaire**. Elle sert seulement si l'estimation automatique de l'objectif
pose problème, par exemple si l'acteur bouge très peu.
`mocapcam damier studio` crée une image `damier.png`. Affichez-la en plein écran et filmez-la
30 s sous tous les angles avec chaque téléphone, puis lancez
`mocapcam intrinseques studio iphone video.mov`. Si ce fichier existe, il est utilisé à la place
de l'estimation automatique.

## 5. Placement des caméras

* Trépieds à **1,2–1,5 m de haut**, à **3–4 m** du centre de la zone.
* Les deux téléphones **devant l'acteur**, écartés de **70 à 90°** (par exemple un à gauche
  et un à droite de la diagonale avant).
* Le corps entier doit rester dans le cadre, **bras levés compris**, et les boules visibles.
* Avec seulement 2 caméras, gardez **les mains face aux caméras**. Une main cachée derrière
  le corps ou derrière l'autre main ne peut pas être reconstruite : l'animation garde alors
  sa dernière position connue.

## 6. Faire une prise avec MocapCam Studio (recommandé)

1. Double-cliquez sur **`MocapCam Studio.bat`**. Une fenêtre noire (le serveur, à laisser ouverte)
   et l'interface s'ouvrent. La première fois, **Windows demande l'autorisation réseau :
   acceptez pour les réseaux privés**, sinon les téléphones ne pourront pas se connecter.
2. Sur chaque téléphone, ouvrez l'app **MocapCam**, donnez-lui son nom (`iphone`, `poco`)
   et **scannez le QR code** affiché dans Studio. Le PC et les téléphones doivent être sur
   le même Wi-Fi, pas un réseau « invité ».
3. Dans Studio, chaque caméra montre son image en direct, son nombre d'images par seconde,
   sa stabilisation (qui doit être *off*), sa batterie et l'état de son horloge. Ensuite :
   * **Identifier** : le nom de la caméra s'affiche en grand sur le téléphone ;
   * **Vérifier les boules** : pour chaque caméra, le nombre de boules vues (il en faut au moins 4)
     et une image de contrôle ;
   * **Bloquer l'exposition**, une fois la lumière réglée. L'option *obturation rapide* (iPhone)
     réduit le flou des mains.
4. **● Enregistrer** : tous les téléphones démarrent en même temps, puis le PC émet un **bip**
   qui sert à les synchroniser à la milliseconde près. Gardez le son du PC allumé et assez fort.
   Pas besoin de clap.
5. Jouez l'action (T-pose 2 s, puis une dizaine de secondes de marche dans la zone si l'action
   est statique), puis **■ Arrêter**. Les vidéos sont envoyées automatiquement au PC, dans
   `captures\prise_XXX\`.
6. Dans la liste **Prises**, cliquez **Traiter**. Le déroulement s'affiche dans le panneau
   *Traitement*. Le résultat est **`captures\prise_XXX\prise_XXX.fbx`** : l'animation sur un
   squelette Mixamo (voir section 8). Le sous-dossier `mocapcam\` contient les données de
   travail (vidéos normalisées, détections, contrôles), inutiles pour l'utilisation.

## 7. Sans l'app : vidéos filmées à la main

Si vous filmez avec l'appareil photo normal des téléphones :
1. Lancez l'enregistrement sur les deux téléphones, puis faites un **clap net et sonore**.
2. Copiez les vidéos en qualité d'origine dans un dossier de prise, nommées `iphone.mov` et `poco.mp4`.
3. Glissez le dossier sur **`Verifier les boules.bat`** pour vérifier, puis sur **`Traiter une prise.bat`**.

La détection 2D est gardée en cache dans `prise\mocapcam\` : un deuxième lancement ne la refait
pas. L'option `--force` refait tout.

**Durée** : la détection 2D est l'étape longue. Sur processeur, comptez plusieurs minutes par
minute de vidéo. L'affinage global du mouvement (corps puis chaque main, sur toute la prise
d'un coup) ajoute environ 5 minutes pour 20 s de prise : c'est lui qui rend le bassin immobile
quand l'acteur ne bouge pas, les os de longueur constante et les doigts sans tressautements. Avec une carte NVIDIA, installez `onnxruntime-gpu` et mettez `"device": "cuda"`
dans `studio.json`. Pour aller plus vite au détriment de la précision : `"mode": "balanced"`.

**Si le son ne synchronise pas** (confiance faible), indiquez le décalage à la main :
`mocapcam traiter studio prise1 --decalage poco=0.125` (en secondes).

## 8. Le FBX et Blender

**Le FBX** (`prise_XXX.fbx`) est créé automatiquement à la fin du traitement, avec Blender
(lancé en arrière-plan, sans fenêtre ; Blender doit être installé, il est trouvé tout seul).
C'est l'équivalent d'une animation Mixamo téléchargée « sans skin » :
* squelette aux **65 os Mixamo** (`mixamorig:Hips`, `mixamorig:LeftHandIndex1`, …), aux
  proportions de l'acteur, animé corps et doigts, une clé par image (60 i/s) ;
* **pose de repos en T-pose** (bras à l'horizontale, paumes vers le bas, jambes droites) :
  c'est la pose que voient Blender, Unity ou Unreal à l'import, sans image ajoutée à l'animation ;
* Y vers le haut, personnage tourné vers +Z, à taille réelle, sans os de bout ajoutés ;
* le personnage part de l'origine (déplacement relatif). Pour garder la position dans la zone
  de capture : `mocapcam fbx captures\prise_XXX --absolu`.

Utilisation :
* **Unity** : importez le FBX, *Rig > Animation Type : Humanoid* (les noms Mixamo sont reconnus
  tout seuls), puis utilisez l'animation sur n'importe quel personnage humanoïde.
* **Unreal** : importez-le sur le squelette du FBX, puis reciblez avec l'IK Retargeter
  (préréglage Mixamo).
* **Blender et autres** : tout outil qui accepte les animations Mixamo (Rokoko, Auto-Rig Pro,
  Expy Kit, Cascadeur…).

Pour refaire un FBX : `mocapcam fbx captures\prise_XXX` (ou `--blender "C:\...\blender.exe"`
si Blender n'est pas trouvé ; le chemin peut aussi être mis dans `studio.json`, clé `blender`).

**Dans Blender, avec l'add-on MocapCam** (barre latérale, touche N > onglet **MocapCam**) :
1. **Animation** : *Importer une animation (.fbx)* (aussi dans *Fichier > Importer > Animation
   MocapCam (.fbx)*). Le squelette animé apparaît, nommé comme la prise, et devient la **Source**.
   La cadence et la plage d'images de la scène sont réglées toutes seules.
2. **Personnage** : choisissez la **Cible**, l'armature de votre personnage (par exemple
   « Armature » pour un personnage téléchargé sur Mixamo), puis cliquez **Retarget**.
   * Chaque os du personnage prend la même direction et la même torsion que dans
     l'animation (rotations identiques image par image, aucun tressautement ajouté) ; le préfixe
     (`mixamorig:`, `mixamorig1:`…), l'échelle 0,01, la rotation de l'objet et la pose de
     repos (T-pose ou A-pose) sont gérés.
   * Le déplacement est mis à l'échelle du personnage. En mode *Relatif* (par défaut), il part
     de sa position actuelle ; en mode *Absolu*, il suit la zone de capture.
   * La source est cachée ensuite (case *Cacher la source*), son animation reste disponible.
   * La scène est remise à la cadence de la capture (ex. 60 i/s). Importer un personnage
     Mixamo (FBX à 30 i/s) passe la scène à 30 i/s et l'animation serait jouée au ralenti ;
     si ça arrive après le retarget, le panneau l'affiche en rouge avec un bouton pour corriger.
3. **Exporter en FBX** : le personnage animé, avec la pose de repos de son armature.

Le retarget marche avec n'importe quelle animation au squelette Mixamo, pas seulement MocapCam.

## 9. Réglages avancés (`studio.json`)

| Clé | Rôle |
|---|---|
| `cameras` | noms des caméras, donc des fichiers vidéo. Ajoutez un 3ᵉ téléphone ici. |
| `marker_height` | hauteur du centre visible des boules : 0,4 × le rayon pour une demi-boule (0,016 m pour une boule de 8 cm coupée en deux) |
| `markers[].position` | positions en mètres. Pour un carré, réglez plutôt le côté dans Studio. |
| `sync_by_motion` | `true` : après le bip, la synchro est affinée avec le mouvement de l'acteur. Dans un téléphone, le son et l'image peuvent être décalés de plusieurs dizaines de ms (100 ms mesurées sur un Poco X3) : sans cette correction, les gestes rapides tremblent. |
| `blender` | chemin de `blender.exe` pour le FBX, si Blender n'est pas trouvé tout seul |
| `fbx` | `false` pour ne pas créer de FBX |
| `markers` | ajouter des boules (6 à 8) améliore la première estimation. Une boule posée sur un objet de hauteur connue (Z ≠ 0) l'améliore beaucoup. |
| `markers[].hsv` | couleur sur mesure `[teinte 0-180, tolérance, saturation min, luminosité min]` |
| `reconstruction.score_hands` | seuil de confiance des mains (augmentez-le si les doigts tremblent) |
| `filtering.cutoff_body_hz` / `cutoff_hands_hz` | lissage de la première estimation (plus bas = plus lisse) |
| `refine.smooth_body` / `smooth_hands` | fluidité imposée par l'affinage global, corps et doigts (plus haut = plus lisse, 2 et 8 par défaut) |
| `refine.enabled` | `false` pour sauter l'affinage global (traitement plus rapide, mouvement moins propre) |
| `refine.knee_min_deg` / `knee_side_deg` | genoux en charnière : flexion minimale (2° : jamais tendus à bloc ni pliés vers l'arrière) et écart maximal de leur direction par rapport au pied (30°) |
| `filtering.max_gap_s` | durée maximale d'un trou comblé par interpolation |

## 10. Dépannage

| Problème | Solution |
|---|---|
| « Boules introuvables ou incohérentes » | Regardez `fond_*.jpg` : moins de 4 boules visibles, boules trop petites, couleur délavée (HDR activé ?), éclairage trop faible (ajustez `hsv`), ou carré mal mesuré. |
| « boules … semblent échangées » | Deux boules ne sont pas à la place du tableau. C'est pris en compte : vous pouvez les laisser ainsi. |
| « taille peu plausible » | Le côté du carré indiqué dans Studio ne correspond pas à la réalité : remesurez-le, enregistrez-le, puis retraitez la prise (rapide : la détection 2D est gardée). |
| Erreur de calibration > 4 px | Normal avant l'affinage par le corps si les boules sont un peu mal placées. Au-delà de 10 px : une boule mal mesurée, ou une tache de même couleur confondue avec une boule. |
| « Format d'image différent » | Seulement avec le damier optionnel : il a été filmé dans une autre orientation ou une autre résolution. |
| Le téléphone ne trouve pas le PC | Même Wi-Fi (pas « invité ») ; autorisez Python dans le pare-feu Windows (réseaux privés) ; essayez l'adresse affichée sous le QR code. |
| Signal sonore « peu audible » | Montez le volume du PC. Sinon, la synchro se fait par les horloges, à environ une image près. |
| « Stabilisation » différente de *off* | Le téléphone ne permet pas de la couper à 60 i/s : réessayez à 30 i/s, ou acceptez une précision un peu moindre. |
| L'app iPhone ne s'ouvre plus | Les 7 jours sont passés : réinstallez-la avec Sideloadly. |
| « FBX non créé : Blender introuvable » | Installez Blender (blender.org), ou indiquez son chemin dans `studio.json`, clé `blender`. |
| Vidéo plus serrée que l'aperçu | Mettez à jour l'app (1.0.5 ou plus). |
| « Pas assez de points communs » | L'acteur est trop peu visible par les deux caméras à la fois. Rapprochez les caméras de l'axe de jeu. |
| Doigts qui tremblent | Rapprochez les caméras, éclairez mieux, baissez `cutoff_hands_hz` à 5–6. |
| Bras qui tressautent dans les gestes rapides | Vérifiez dans le journal la ligne « synchro fine par le mouvement » ; montez la lumière (moins de flou de bougé) et bloquez l'exposition avec l'obturation rapide. |
| Bras ou jambe qui « saute » | Membre caché pour une caméra : ajoutez un 3ᵉ téléphone, ou orientez l'action face aux caméras. |

## Tests

* `python tests/test_synthetique.py` (nécessite `pip install bpy`) simule une capture complète :
  bruit, inversions gauche/droite, mains cachées, personne parasite. Il vérifie chaque étape
  jusqu'au résultat dans Blender, sur trois types de squelettes.
* `python tests/test_studio.py` (nécessite `pip install playwright`) pilote MocapCam Studio dans un
  navigateur avec deux faux téléphones : connexion, vérification des boules, enregistrement, bip,
  rapatriement et synchronisation.

Le code de l'app mobile est dans `mobile\MocapCamPhone` (Expo + VisionCamera 5). `PROTOCOLE.md`
décrit les échanges entre le PC et les téléphones.
