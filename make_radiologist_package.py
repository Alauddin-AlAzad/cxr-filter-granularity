"""
Builds the folder that goes to the radiologist.

Takes the 50 cases incxr_eval.py selected, copies the images out under neutral
names, and writes a self-contained rating sheet that works offline from the
file system and exports the ratings as CSV.

The normal/abnormal label is NOT in the package. It stays in
incxr_radiologist_key.csv on this machine, so the reading is blind.

    python make_radiologist_package.py
    python make_radiologist_package.py --single
    python make_radiologist_package.py --max_side 2000 --out package_v2

--single writes one HTML file with the images embedded, for a reader who
cannot unzip a folder. It is large but needs nothing but a browser.

Images are IN-CXR, not MIMIC, so the package can be shared outside PhysioNet.
"""

import argparse
import base64
import csv
import io
import json
import os
import shutil

import pandas as pd
from PIL import Image

CASES = "incxr_for_radiologist.csv"
OUT = "radiologist_package"
MAX_SIDE = 1600
QUALITY = 92


def prepare(src, max_side):
    """Returns the jpeg bytes to use, and whether it had to be resized."""
    im = Image.open(src)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("L").save(buf, "JPEG", quality=QUALITY)
        return buf.getvalue(), True
    with open(src, "rb") as f:
        return f.read(), False


def data_uri(raw):
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def write_csv_fallback(path, cases):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "generated_report", "report_correct_1_5",
                    "completeness_1_5", "fluency_1_5", "your_read_normal_abnormal",
                    "would_you_sign_yes_no", "what_it_missed"])
        for c in cases:
            w.writerow([c["case"], c["report"], "", "", "", "", "", ""])


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chest X-ray Report Rating</title>
<style>
  :root{
    --ink:#16181d; --mute:#6c7280; --line:#dfe2e8; --soft:#f5f6f8;
    --accent:#1f5f8b; --warn:#9a3a2a; --ok:#2f6b4f; --bg:#fff;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --ink:#e8eaee; --mute:#9aa1ab; --line:#333942; --soft:#1c2026;
      --accent:#6fb3dd; --warn:#e08b76; --ok:#7fc3a0; --bg:#131519;
    }
  }
  :root[data-theme="dark"]{
    --ink:#e8eaee; --mute:#9aa1ab; --line:#333942; --soft:#1c2026;
    --accent:#6fb3dd; --warn:#e08b76; --ok:#7fc3a0; --bg:#131519;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{position:sticky;top:0;z-index:5;background:var(--bg);
    border-bottom:1px solid var(--line);padding:10px 16px;
    display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:15px;margin:0;font-weight:650;letter-spacing:-.2px}
  .bar{flex:1;min-width:120px;height:5px;background:var(--soft);border-radius:3px;overflow:hidden}
  .bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .2s}
  .count{font-variant-numeric:tabular-nums;color:var(--mute);font-size:13px}
  button{font:inherit;padding:7px 14px;border:1px solid var(--line);
    background:var(--bg);color:var(--ink);border-radius:7px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.pri{background:var(--accent);border-color:var(--accent);color:#fff}
  main{max-width:1180px;margin:0 auto;padding:16px}
  .intro{background:var(--soft);border-left:3px solid var(--accent);
    padding:14px 16px;border-radius:0 8px 8px 0;margin-bottom:18px;font-size:14px}
  .intro ol{margin:8px 0 0;padding-left:20px}
  .wrap{display:grid;grid-template-columns:1.25fr 1fr;gap:20px;align-items:start}
  @media (max-width:860px){.wrap{grid-template-columns:1fr}}
  .imgbox{background:#000;border-radius:9px;overflow:hidden;
    display:flex;align-items:center;justify-content:center;min-height:300px}
  .imgbox img{width:100%;height:auto;display:block}
  .rep{background:var(--soft);border:1px solid var(--line);border-radius:9px;
    padding:13px 15px;font-size:15px;line-height:1.65;margin-bottom:16px}
  .rep .lbl{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
    color:var(--mute);margin-bottom:5px}
  .q{margin-bottom:15px}
  .q p{margin:0 0 6px;font-weight:600;font-size:14px}
  .q .hint{font-weight:400;color:var(--mute);font-size:12.5px;display:block;margin-top:1px}
  .opts{display:flex;gap:6px;flex-wrap:wrap}
  .opts label{border:1px solid var(--line);border-radius:7px;padding:6px 12px;
    cursor:pointer;font-size:14px;user-select:none;min-width:38px;text-align:center}
  .opts input{position:absolute;opacity:0;width:0;height:0}
  .opts input:checked + span{color:#fff}
  .opts label:has(input:checked){background:var(--accent);border-color:var(--accent);color:#fff}
  textarea{width:100%;min-height:62px;font:inherit;padding:9px 11px;
    border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);resize:vertical}
  .nav{display:flex;gap:9px;align-items:center;margin-top:18px;flex-wrap:wrap}
  .saved{color:var(--ok);font-size:13px}
  .note{font-size:12.5px;color:var(--mute);margin-top:14px}
</style>
</head>
<body>
<header>
  <h1>Chest X-ray Report Rating</h1>
  <div class="bar"><i id="prog"></i></div>
  <span class="count" id="count"></span>
  <button id="exp" class="pri">Export CSV</button>
</header>

<main>
  <div class="intro" id="intro">
    <strong>What this is.</strong> Fifty chest radiographs, each with a report written
    automatically by a research model. Please read the image, then judge the report.
    <ol>
      <li>The radiographs are from a public Indian screening dataset. No patient identifiers are present.</li>
      <li>Normal and abnormal cases are mixed, in random order, and not labelled.</li>
      <li>Answers are stored in this browser as you go. Press <em>Export CSV</em> at the
          end &mdash; or at any point &mdash; and send back the downloaded file.</li>
      <li>Leave anything blank if you would rather not answer it.</li>
    </ol>
  </div>

  <div class="wrap">
    <div class="imgbox"><img id="img" alt="chest radiograph"></div>
    <div>
      <div class="rep"><div class="lbl">Model-generated report</div><span id="rep"></span></div>
      <div id="qs"></div>
      <div class="nav">
        <button id="prev">&larr; Previous</button>
        <button id="next" class="pri">Next &rarr;</button>
        <span class="saved" id="saved"></span>
      </div>
      <p class="note">Nothing is uploaded anywhere. The answers stay in this browser
      until you export them.</p>
    </div>
  </div>
</main>

<script>
const CASES = __CASES__;
const KEY = "cxr_ratings_v1";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }

const QS = [
  {id:"correct", label:"Is what the report states correct for this image?",
   hint:"1 = mostly wrong, 5 = everything stated is correct",
   opts:["1","2","3","4","5"]},
  {id:"complete", label:"Does it cover the important findings?",
   hint:"1 = misses nearly everything, 5 = nothing important left out",
   opts:["1","2","3","4","5"]},
  {id:"fluency", label:"Does it read like a radiology report?",
   hint:"1 = incoherent, 5 = reads naturally",
   opts:["1","2","3","4","5"]},
  {id:"read", label:"Your own reading of the image",
   hint:"independent of the report",
   opts:["Normal","Abnormal","Unsure"]},
  {id:"sign", label:"Would you sign this report as written?",
   hint:"after your usual edits, would it be usable",
   opts:["Yes","No"]}
];

let i = 0;

function save(){
  try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
  const s = document.getElementById("saved");
  s.textContent = "saved";
  clearTimeout(save._t);
  save._t = setTimeout(() => { s.textContent = ""; }, 1200);
}

function done(){
  return CASES.filter(c => { const a = answers[c.case]; return a && a.correct; }).length;
}

function render(){
  const c = CASES[i];
  const a = answers[c.case] || (answers[c.case] = {});
  document.getElementById("img").src = c.file;
  document.getElementById("rep").textContent = c.report;
  document.getElementById("count").textContent =
    "case " + (i + 1) + " of " + CASES.length + "  ·  " + done() + " rated";
  document.getElementById("prog").style.width = (100 * done() / CASES.length) + "%";
  document.getElementById("intro").style.display = i === 0 ? "" : "none";

  const box = document.getElementById("qs");
  box.innerHTML = "";
  QS.forEach(q => {
    const d = document.createElement("div");
    d.className = "q";
    const p = document.createElement("p");
    p.textContent = q.label;
    const h = document.createElement("span");
    h.className = "hint";
    h.textContent = q.hint;
    p.appendChild(h);
    d.appendChild(p);
    const row = document.createElement("div");
    row.className = "opts";
    q.opts.forEach(o => {
      const lab = document.createElement("label");
      const inp = document.createElement("input");
      inp.type = "radio";
      inp.name = q.id;
      inp.value = o;
      inp.checked = a[q.id] === o;
      inp.addEventListener("change", () => { a[q.id] = o; save(); render_counts(); });
      const sp = document.createElement("span");
      sp.textContent = o;
      lab.appendChild(inp);
      lab.appendChild(sp);
      row.appendChild(lab);
    });
    d.appendChild(row);
    box.appendChild(d);
  });

  const d = document.createElement("div");
  d.className = "q";
  const p = document.createElement("p");
  p.textContent = "Anything the report missed or got wrong?";
  const h = document.createElement("span");
  h.className = "hint";
  h.textContent = "optional, a few words is enough";
  p.appendChild(h);
  d.appendChild(p);
  const ta = document.createElement("textarea");
  ta.value = a.missed || "";
  ta.addEventListener("input", () => { a.missed = ta.value; save(); });
  d.appendChild(ta);
  box.appendChild(d);

  document.getElementById("prev").disabled = i === 0;
  document.getElementById("next").disabled = i === CASES.length - 1;
}

function render_counts(){
  document.getElementById("count").textContent =
    "case " + (i + 1) + " of " + CASES.length + "  ·  " + done() + " rated";
  document.getElementById("prog").style.width = (100 * done() / CASES.length) + "%";
}

document.getElementById("prev").onclick = () => { if (i > 0) { i--; render(); window.scrollTo(0,0); } };
document.getElementById("next").onclick = () => { if (i < CASES.length - 1) { i++; render(); window.scrollTo(0,0); } };

document.getElementById("exp").onclick = () => {
  const head = ["case","image","correct_1_5","completeness_1_5","fluency_1_5",
                "your_read","would_sign","missed_or_wrong","generated_report"];
  const esc = v => '"' + String(v === undefined ? "" : v).replace(/"/g, '""') + '"';
  const rows = [head.join(",")];
  CASES.forEach(c => {
    const a = answers[c.case] || {};
    rows.push([c.case, c.name, a.correct, a.complete, a.fluency,
               a.read, a.sign, a.missed, c.report].map(esc).join(","));
  });
  const blob = new Blob(["\\ufeff" + rows.join("\\n")], {type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "cxr_ratings.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
};

render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max_side", type=int, default=MAX_SIDE)
    ap.add_argument("--single", action="store_true",
                    help="one self-contained HTML file instead of a folder")
    args = ap.parse_args()

    if not os.path.isfile(args.cases):
        print("not found: %s  (run incxr_eval.py first)" % args.cases)
        return

    df = pd.read_csv(args.cases)
    need = {"case", "local_path", "generated_report"}
    if not need.issubset(df.columns):
        print("missing columns: %s" % ", ".join(sorted(need - set(df.columns))))
        return

    img_dir = os.path.join(args.out, "images")
    if not args.single:
        os.makedirs(img_dir, exist_ok=True)
    else:
        os.makedirs(args.out, exist_ok=True)

    cases, missing, shrunk = [], 0, 0
    for _, r in df.iterrows():
        src = str(r["local_path"])
        if not os.path.isfile(src):
            missing += 1
            continue
        try:
            raw, resized = prepare(src, args.max_side)
        except Exception as e:
            print("  unreadable: %s (%s)" % (os.path.basename(src), type(e).__name__))
            missing += 1
            continue
        shrunk += resized
        name = "case_%02d.jpg" % int(r["case"])
        if args.single:
            ref = data_uri(raw)
        else:
            with open(os.path.join(img_dir, name), "wb") as f:
                f.write(raw)
            ref = "images/" + name
        cases.append({"case": int(r["case"]), "file": ref, "name": name,
                      "report": str(r["generated_report"]).strip()})

    if not cases:
        print("no images could be read")
        return

    page = PAGE.replace("__CASES__", json.dumps(cases, ensure_ascii=False))
    name = "rating_sheet_one_file.html" if args.single else "rating_sheet.html"
    sheet = os.path.join(args.out, name)
    with open(sheet, "w", encoding="utf-8") as f:
        f.write(page)
    write_csv_fallback(os.path.join(args.out, "rating_sheet_paper_version.csv"), cases)

    mb = os.path.getsize(sheet) / 1048576.0
    print("cases: %d   resized: %d   missing: %d" % (len(cases), shrunk, missing))
    print("file:   %s   (%.1f MB)" % (os.path.abspath(sheet), mb))
    if args.single:
        print("send this one file; nothing to unzip")
        if mb > 60:
            print("WARNING: large. rerun with --max_side 1200 for a lighter file")
    else:
        print("folder: %s" % os.path.abspath(args.out))
    print("\nthe normal/abnormal key stays in incxr_radiologist_key.csv - do not send it")


if __name__ == "__main__":
    main()