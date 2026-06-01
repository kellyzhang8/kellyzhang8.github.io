from build_pmtiles_fast import write_project_year, DATA_DIR, PROJECTS, YEARS

for project in PROJECTS:
    for year in YEARS:
        h = DATA_DIR / f"{project}_{year}_height.pmtiles"
        l = DATA_DIR / f"{project}_{year}_landtype.pmtiles"
        if h.exists() and h.stat().st_size > 1024 and l.exists() and l.stat().st_size > 1024:
            continue
        write_project_year(project, year)
