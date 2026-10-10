import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { execFileSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const script = JSON.parse(readFileSync(join(ROOT, "script.json"), "utf-8"));
const SCREENS = join(ROOT, "screenshots");

const B = script.brand;
// "mobile" -> narrow phone-frame media, side-by-side with copy.
// "web" -> wide browser-chrome-frame media, stacked above the copy.
const PLATFORM = script.platform === "web" ? "web" : "mobile";

// CUSTOMIZE: for each scene id in script.json, optionally provide a richer
// written kicker/body/bullets for the PDF (beyond the spoken narration).
// Any scene id left out just falls back to scene.narration as the body with
// no bullet list — so this can start as {} and be filled in per app.
const DETAILS = JSON.parse(readFileSync(join(__dirname, "details.json"), "utf-8"));

// CUSTOMIZE: path to the target app's real logo file (PNG/SVG with a
// transparent or light background — it renders inside a white pill badge,
// see .mark CSS below). Copy the app's actual logo asset into
// walkthrough/assets/logo.png and reference it here, e.g.:
//   const LOGO_PATH = join(ROOT, "assets", "logo.png");
const LOGO_PATH = join(ROOT, "assets", "logo.png");
const fmtDate = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

// A scene's "redact" rects ([x, y, w, h] as fractions of the screenshot) are blurred in a copy of the
// image, so personal data hidden in the video is hidden in the PDF too.
function redacted(scene) {
  const src = join(SCREENS, scene.image);
  if (!scene.redact || !scene.redact.length) return src;
  const buf = readFileSync(src), W = buf.readUInt32BE(16), H = buf.readUInt32BE(20);
  const dir = join(SCREENS, ".redacted");
  mkdirSync(dir, { recursive: true });
  const out = join(dir, scene.image);
  let chain = "", last = "0:v";
  scene.redact.forEach(([x, y, w, h], k) => {
    const [px, py, pw, ph] = [x * W, y * H, w * W, h * H].map(Math.round);
    chain += `[0:v]crop=${pw}:${ph}:${px}:${py},gblur=sigma=14[b${k}];[${last}][b${k}]overlay=${px}:${py}[o${k}];`;
    last = `o${k}`;
  });
  execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-i", src, "-filter_complex", chain.slice(0, -1), "-map", `[${last}]`, out]);
  return out;
}

const sectionsHtml = script.scenes
  .map((scene, i) => {
    const d = DETAILS[scene.id] || {};
    const imgPath = scene.image ? redacted(scene) : null;
    const num = String(i + 1).padStart(2, "0");
    const media =
      PLATFORM === "web"
        ? imgPath
          ? `<div class="browser-frame"><div class="browser-bar"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>${scene.url ? `<span class="browser-url">${scene.url}</span>` : ""}</div><img src="file://${imgPath}" /></div>`
          : ""
        : imgPath
          ? `<div class="phone-frame"><img src="file://${imgPath}" /></div>`
          : `<div class="phone-frame placeholder"></div>`;
    return `
  <div class="page section-page">
    <section class="scene ${PLATFORM === "web" ? "web" : i % 2 === 1 ? "reverse" : ""}${media ? "" : " text-only"}">
      ${media ? `<div class="scene-media">${media}</div>` : ""}
      <div class="scene-copy">
        <div class="scene-num">${num} / ${String(script.scenes.length).padStart(2, "0")}</div>
        <div class="kicker">${d.kicker || scene.eyebrow}</div>
        <h2>${scene.title}</h2>
        <p class="lead">${d.body || scene.narration}</p>
        ${
          d.bullets
            ? `<ul>${d.bullets.map((b) => `<li>${b}</li>`).join("")}</ul>`
            : ""
        }
      </div>
    </section>
    <div class="footer-strip"><span>${script.appName}</span><span>${num} / ${String(script.scenes.length).padStart(2, "0")} — ${scene.title}</span></div>
  </div>`;
  })
  .join("\n");

const tocHtml = script.scenes
  .map((scene, i) => {
    const num = String(i + 1).padStart(2, "0");
    const d = DETAILS[scene.id] || {};
    return `<div class="toc-row"><span class="toc-num">${num}</span><span class="toc-title">${scene.title}</span><span class="toc-kicker">${d.kicker || scene.eyebrow}</span></div>`;
  })
  .join("\n");

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>${script.appName} — System Walkthrough</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --primary: ${B.primary};
    --primary-light: ${B.primaryLight};
    --primary-dark: ${B.primaryDark};
    --gold: ${B.gold};
    --on-dark: ${B.silver};
    --rule: #d2d2cc;
    --bg: ${B.bg};
    --surface: ${B.surface};
    --text: ${B.textPrimary};
    --text-secondary: ${B.textSecondary};
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: 'Instrument Sans', 'Avenir Next', sans-serif;
    color: var(--text);
    background: var(--bg);
    font-size: 12.5px;
  }
  h1, h2, h3 { font-family: 'Instrument Sans', 'Avenir Next', sans-serif; margin: 0; }
  @page { size: A4; margin: 0; }

  .page { width: 210mm; height: 297mm; position: relative; page-break-after: always; page-break-inside: avoid; overflow: hidden; }
  .page:last-child { page-break-after: auto; }

  /* Cover */
  .cover {
    background: radial-gradient(circle at 50% 15%, var(--primary-light) 0%, var(--primary) 55%, var(--primary-dark) 100%);
    color: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40mm 20mm;
  }
  .mark { margin-bottom: 14mm; display: inline-flex; }
  .mark img { height: 24mm; display: block; border-radius: 5mm; border: 1px solid rgba(236,236,232,0.28); }
  .mark.small { margin-bottom: 8mm; }
  .mark.small img { height: 16mm; border-radius: 3.4mm; }
  .cover .eyebrow {
    font-family: 'Instrument Sans', 'Avenir Next', sans-serif;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--on-dark);
    font-size: 12px;
    margin-bottom: 8mm;
  }
  .cover h1 {
    font-weight: 500;
    font-size: 50px;
    letter-spacing: -1.8px;
    line-height: 1.15;
    margin-bottom: 6mm;
  }
  .cover .tagline {
    font-size: 16px;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
    margin-bottom: 26mm;
  }
  .cover .meta {
    font-size: 11px;
    color: rgba(255,255,255,0.6);
    letter-spacing: 1px;
  }
  .cover .divider { width: 48px; height: 2px; background: var(--on-dark); margin: 10mm 0; }

  /* TOC */
  .toc-page { padding: 18mm 22mm; background: var(--surface); }
  .toc-page .kicker-top {
    font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: var(--primary); font-size: 11px; margin-bottom: 4mm;
  }
  .toc-page h1 { font-size: 26px; font-weight: 500; letter-spacing: -0.6px; color: var(--text); margin-bottom: 7mm; }
  .toc-row {
    display: flex; align-items: baseline; gap: 6mm;
    padding: 1.75mm 0; border-bottom: 1px solid var(--rule);
  }
  .toc-num { font-family: 'Instrument Sans', 'Avenir Next', sans-serif; font-weight: 600; color: var(--gold); font-size: 11px; width: 10mm; font-variant-numeric: tabular-nums; }
  .toc-title { font-family: 'Instrument Sans', 'Avenir Next', sans-serif; font-weight: 500; font-size: 12px; flex: 1; color: var(--text); }
  .toc-kicker { font-size: 8.5px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }

  /* Section pages */
  .section-page { background: var(--surface); padding: 16mm 16mm; }
  .scene {
    display: flex;
    align-items: center;
    gap: 12mm;
    height: 265mm;
  }
  .scene.reverse { flex-direction: row-reverse; }
  .scene-media { flex: 0 0 68mm; display: flex; justify-content: center; }
  .phone-frame {
    width: 62mm;
    border-radius: 7mm;
    background: #0B0F0C;
    padding: 2.2mm;
    box-shadow: 0 10px 30px -8px rgba(18,32,24,0.4);
  }
  .phone-frame img { width: 100%; display: block; border-radius: 5.2mm; }
  .phone-frame.placeholder { height: 130mm; }

  /* Web platform: stacked layout, wide browser-chrome frame on top */
  .scene.web { flex-direction: column; align-items: stretch; gap: 8mm; height: auto; }
  .scene.text-only { padding-top: 14mm; }
  .scene.text-only .lead { font-size: 15px; }
  .scene.text-only li { font-size: 13px; margin-bottom: 3.6mm; }
  code { font-family: "SF Mono", Menlo, monospace; font-size: 0.92em; background: #e6e6e4; padding: 0.2mm 1.2mm; border-radius: 1mm; }
  li b { color: var(--text); font-weight: 600; }
  .scene.web .scene-media { flex: none; width: 100%; }
  .browser-frame {
    width: 100%; max-height: 150mm; border-radius: 4mm; overflow: hidden;
    background: #d2d2cc; box-shadow: 0 8px 24px -10px rgba(0,0,0,0.35); border: 1px solid var(--rule);
    display: flex; flex-direction: column;
  }
  /* object-fit: contain — a web screenshot's aspect ratio is unpredictable
     (a 16:9 viewport capture vs. a tall full-page capture), so this always
     letterboxes inside the capped height instead of blowing out the page. */
  .browser-frame img { width: 100%; flex: 1; min-height: 0; object-fit: contain; background: #e6e6e4; display: block; }
  .browser-frame.placeholder { height: 90mm; }
  .browser-bar { display: flex; align-items: center; gap: 3mm; padding: 2.2mm 4mm; background: #dcdcd8; }
  .browser-bar .dot { width: 2.6mm; height: 2.6mm; border-radius: 50%; display: inline-block; }
  .browser-bar .dot.r { background: #FF5F57; }
  .browser-bar .dot.y { background: #FEBC2E; }
  .browser-bar .dot.g { background: #28C840; }
  .browser-url {
    margin-left: 2mm; background: #eeeeec; border-radius: 999px; padding: 1.3mm 4mm;
    font-size: 9px; color: var(--text-secondary); font-weight: 600;
  }

  .scene-copy { flex: 1; }
  .scene-num {
    font-family: 'Instrument Sans', 'Avenir Next', sans-serif; font-weight: 600; color: var(--gold);
    font-size: 11px; letter-spacing: 1px; margin-bottom: 3mm;
  }
  .kicker {
    font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: var(--text-secondary); font-size: 9.5px; margin-bottom: 3mm;
  }
  .scene-copy h2 { font-size: 27px; font-weight: 500; letter-spacing: -0.6px; color: var(--text); margin-bottom: 5mm; line-height: 1.2; }
  .scene-copy .lead { font-size: 14px; line-height: 1.6; color: var(--text); margin-bottom: 6mm; }
  .scene-copy ul { margin: 0; padding-left: 4.5mm; }
  .scene-copy li { font-size: 12.5px; line-height: 1.55; color: var(--text-secondary); margin-bottom: 3mm; }
  .scene-copy li::marker { color: var(--gold); }

  /* Closing */
  .closing {
    background: radial-gradient(circle at 50% 85%, var(--primary-light) 0%, var(--primary) 55%, var(--primary-dark) 100%);
    color: #fff; display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center; padding: 40mm 24mm;
  }
  .closing h2 { font-size: 30px; font-weight: 500; letter-spacing: -0.8px; margin-bottom: 6mm; }
  .closing p { font-size: 13px; color: rgba(255,255,255,0.8); max-width: 130mm; line-height: 1.7; }
  .closing .tag { color: var(--on-dark); font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: 11px; margin-top: 10mm; }

  .footer-strip {
    position: absolute; bottom: 10mm; left: 16mm; right: 16mm;
    display: flex; justify-content: space-between;
    font-size: 9px; color: var(--text-secondary); letter-spacing: 0.5px;
  }
</style>
</head>
<body>

  <div class="page cover">
    <div class="mark"><img src="file://${LOGO_PATH}" /></div>
    <div class="eyebrow">Complete System Walkthrough</div>
    <h1>${script.appName}</h1>
    <div class="tagline">${script.tagline}</div>
    <div class="divider"></div>
    <div class="meta">Feature Guide &amp; Screen-by-Screen Reference · ${fmtDate}</div>
  </div>

  <div class="page toc-page">
    <div class="kicker-top">Contents</div>
    <h1>Everything in this guide</h1>
    ${tocHtml}
    <div class="footer-strip"><span>${script.appName}</span><span>Page 2</span></div>
  </div>

  ${sectionsHtml}

  <div class="page closing">
    <div class="mark small"><img src="file://${LOGO_PATH}" /></div>
    <h2>Thank you</h2>
    <p>This guide covered every page of ${script.appName}: starting an analysis, reading a run from verdict to audit trail, human vetting, the atlas and the map, the benchmarks that score the system on real stars, and the automation, settings and method behind it.</p>
    <div class="tag">${script.tagline}</div>
  </div>

</body>
</html>`;

writeFileSync(join(ROOT, "docs", "walkthrough.html"), html);
console.log("wrote docs/walkthrough.html —", script.scenes.length, "sections");
