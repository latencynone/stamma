# Stämifier

Sjung in en melodi (max 10 sekunder), eller ladda upp en ljudfil — appen känner av tonarten och bygger en stämma (ters/kvint/sext, under/över) i din egen röst. Valfri lätt autotune rättar falska toner i originalet.

## Arkitektur

- **Tonhöjdsdetektering, tonartsdetektering, notsegmentering, stämteori** ([src/pitchAnalysis.js](src/pitchAnalysis.js), [src/musicTheory.js](src/musicTheory.js)) — tidsdomän-autokorrelation, Krumhansl–Schmuckler-tonartsdetektering, diatonisk kvantisering. Körs antingen live under inspelning eller offline mot en uppladdad/decodead ljudfil (`extractFramesFromBuffer`).
- **Stämmotor** ([src/harmonyEngine.js](src/harmonyEngine.js)) — bygger en kontinuerlig pitch-ratio-kurva över hela inspelningen (`buildRatioCurveFromSegments`) och kör den genom [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) (WASM, formantbevarande) i en enda sammanhängande offline-rendering. Ingen klipp-och-klistra av enskilda toner — se kommentarerna i filen för varför (bibliotekets `schedule()`-API är byggt för progressiv/live-schemaläggning, drivs här via `OfflineAudioContext.suspend()/resume()`). Samma kurvbyggare driver både stämman (`buildRatioCurve`, mål = fast intervall) och autotune (`buildAutotuneRatioCurve`, mål = delvis rättning mot närmaste diatoniska ton).
- Gap-hantering mellan toner avgörs av faktisk ljudenergi i inspelningen (`computeEnergyEnvelope`/`hasAudibleEnergy`), inte bara förfluten tid — annars riskerar tystare, otydligt tonhöjdsspårade partier att spelas upp helt oskiftade.

## Kom igång

```bash
npm install
npm run dev
```

Kräver HTTPS eller `localhost` (AudioWorklet/mikrofon).

## Bygg

```bash
npm run build
```

## Deploy

GitHub Actions-workflow i [.github/workflows/deploy.yml](.github/workflows/deploy.yml) bygger och publicerar till GitHub Pages vid push till `main`. Kräver att Pages är aktiverat för repot (Settings → Pages → Source: GitHub Actions). Om appen inte ligger på reposets root-domän, sätt `base` i [vite.config.js](vite.config.js) till `/<repo-namn>/`.
