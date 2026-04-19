# Kelly Zhang's Personal Homepage

A clean, academic personal homepage in the **minimal-light** style, built with Jekyll and served on GitHub Pages.

Live site (after deploying): **https://kellyzhang8.github.io**

---

## 🚀 Quick Deploy (5 minutes, no terminal required)

1. Open your existing repo on GitHub: <https://github.com/kellyzhang8/kellyzhang8.github.io>
2. Click **Add file → Upload files**.
3. Drag **every file and folder** from this project into the upload box
   (make sure the hidden `.gitignore` is included).
4. Scroll down, write a short commit message like `Initial minimal-light homepage`, and click **Commit changes**.
5. Go to **Settings → Pages** (in the repo). Under "Build and deployment", make sure the source is set to **Deploy from a branch → `main` → `/ (root)`**.
6. Wait ~60 seconds. Your site will be live at **https://kellyzhang8.github.io** 🎉

> ⚠️  Your existing `index.html` in the repo will be overwritten — that's expected.

---

## 🧰 Deploy with Git (recommended for ongoing edits)

```bash
# 1) Clone your repo somewhere on your computer
git clone https://github.com/kellyzhang8/kellyzhang8.github.io.git
cd kellyzhang8.github.io

# 2) Copy every file from this project into the repo folder
#    (on macOS / Linux, from the folder where Claude generated the site):
cp -r /path/to/this/kellyzhang8.github.io/* .
cp /path/to/this/kellyzhang8.github.io/.gitignore .

# 3) Commit and push
git add .
git commit -m "Initial minimal-light homepage"
git push origin main
```

Then visit **https://kellyzhang8.github.io** — GitHub Pages will rebuild automatically each time you push.

---

## ✏️ What to customize

Open these files in any text editor (or directly on GitHub.com by clicking the pencil icon):

| File | What to change |
|---|---|
| **`_config.yml`** | Your name, title, affiliation, email, office, and social links |
| **`index.html`** | The "About" paragraph (introduce yourself + research interests) |
| **`_data/news.yml`** | Add/remove news entries — newest on top |
| **`_data/education.yml`** | Your degrees, schools, majors, advisor |
| **`_data/research.yml`** | Your projects — title, description, links |
| **`_data/publications.yml`** | Your papers — title, authors, venue, links |
| **`assets/img/profile.jpg`** | Drop in a square photo (~500×500 px). Then update `photo:` in `_config.yml` |
| **`assets/files/CV_KellyZhang.pdf`** | Put your CV PDF here so the "Curriculum Vitae" button works |

Everything else (`_layouts/`, `_includes/`, `assets/css/`) controls the look — you can ignore those unless you want to tweak the design.

---

## 🖥️ (Optional) Preview locally before pushing

Requires **Ruby 3.0+** installed.

```bash
cd kellyzhang8.github.io
bundle install          # one-time, installs Jekyll + plugins
bundle exec jekyll serve
```

Then open <http://localhost:4000>. Changes to any file (except `_config.yml`) will live-reload.

---

## 🎨 Design notes

- **Palette:** soft off-white background (`#fdfdfc`), slate text, muted teal accent (`#3a7d7b`).
- **Typography:** body uses Inter (UI) + Lora (headings). Served from Google Fonts.
- **Layout:** sticky profile card on the left, scrolling content on the right — collapses to a single column under 860 px.
- **No JavaScript frameworks** — pure Jekyll, pure CSS. Loads instantly.

Want a different accent color? Open `assets/css/main.css` and change the `--accent` variable at the very top.

---

## 🗂️ Project structure

```
.
├── _config.yml               # site-wide settings (edit this first!)
├── index.html                # the single-page homepage
├── _layouts/
│   └── default.html          # <html> shell, loads fonts + stylesheet
├── _includes/
│   ├── header.html           # sticky top nav
│   ├── footer.html           # small footer
│   └── social.html           # social-icon links block
├── _data/
│   ├── news.yml              # short announcements
│   ├── education.yml         # degrees & schools
│   ├── research.yml          # projects with thumbnails
│   └── publications.yml      # papers list
├── assets/
│   ├── css/main.css          # all styles
│   ├── img/profile.svg       # placeholder avatar
│   └── files/                # drop your CV PDF here
├── Gemfile                   # Ruby dependency file for local preview
├── .gitignore
└── README.md                 # this file
```

---

Made with ❤️ in the *minimal-light* style.
