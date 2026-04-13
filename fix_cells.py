import json, uuid

with open('engle_xmatch.ipynb') as f:
    nb = json.load(f)

def make_code_cell(source, cell_id=None):
    return {
        'cell_type': 'code',
        'id': cell_id or uuid.uuid4().hex[:8],
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source
    }

# ── Replace cell 76 (7fecb96d): split missing stars into two groups ─────────
cell76_source = '\n'.join([
    '# Split missing stars into two groups:',
    '#   1. Known Gaia ID (from SIMBAD) but direct query failed → retry by ID',
    '#   2. No Gaia ID → cone search by SIMBAD coordinates',
    'missing_mask = df_final["source_id"].isna()',
    '',
    'retry_direct = df_final[',
    '    missing_mask &',
    '    df_final["gaia_dr3_source_id_from_simbad"].notna()',
    '].drop_duplicates(subset=["gaia_dr3_source_id_from_simbad"])',
    '',
    'needs_cone = df_final[',
    '    missing_mask &',
    '    df_final["gaia_dr3_source_id_from_simbad"].isna() &',
    '    df_final["simbad_ra_deg"].notna()',
    '].drop_duplicates(subset=["simbad_ra_deg", "simbad_dec_deg"])',
    '',
    'print(f"Have Gaia ID, direct query failed → retry: {len(retry_direct)}")',
    'print(f"No Gaia ID, have SIMBAD coords → cone search: {len(needs_cone)}")',
    'print(f"No SIMBAD match at all (unrecoverable): {df_final[missing_mask & df_final[\'simbad_ra_deg\'].isna()][\'star_name\'].nunique()}")',
])

# ── Replace cell 77 (4348e058): retry direct queries by source ID ────────────
cell77_source = '\n'.join([
    'import time',
    '',
    '# Step 1: retry direct Gaia query by source_id for the 73 stars',
    '# whose IDs we already know from SIMBAD (previous attempts hit API errors).',
    'retry_core_rows = []',
    'retry_ap_rows   = []',
    'retry_id_failures = []',
    '',
    'for i, row in enumerate(retry_direct.itertuples(), start=1):',
    '    sid = str(row.gaia_dr3_source_id_from_simbad)',
    '    print(f"{i}/{len(retry_direct)}: {row.star_name}  (id={sid})")',
    '    try:',
    '        c = query_gaia_one_source_id(sid)',
    '        if len(c) > 0:',
    '            retry_core_rows.append(c)',
    '        a = query_ap_one_source_id(sid)',
    '        if len(a) > 0:',
    '            retry_ap_rows.append(a)',
    '        time.sleep(1.0)',
    '    except Exception as e:',
    '        retry_id_failures.append({"star_name": row.star_name, "source_id": sid, "reason": str(e)})',
    '        print(f"  -> ERROR: {e}")',
    '        time.sleep(2.0)',
    '',
    'print(f"\\nDirect retry: {len(retry_core_rows)} recovered, {len(retry_id_failures)} still failing")',
])

# ── New cell 77b: assemble direct retry results and fill df_final ────────────
cell77b_source = '\n'.join([
    '# Assemble direct retry results and fill into df_final.',
    'if retry_core_rows:',
    '    retry_core = pd.concat(retry_core_rows, ignore_index=True)',
    '    retry_ap   = pd.concat(retry_ap_rows,   ignore_index=True) if retry_ap_rows else pd.DataFrame()',
    '',
    '    if len(retry_ap) > 0:',
    '        retry_gaia = retry_core.merge(retry_ap, how="left", on="source_id")',
    '    else:',
    '        retry_gaia = retry_core.copy()',
    '',
    '    retry_gaia["g_rp"]    = retry_gaia["phot_g_mean_mag"] - retry_gaia["phot_rp_mean_mag"]',
    '    retry_gaia["bp_rp_0"] = retry_gaia["bp_rp"] - retry_gaia["ebpminrp_gspphot"]',
    '    retry_gaia["source_id"] = retry_gaia["source_id"].astype("string")',
    '',
    '    gaia_fill_cols = [',
    '        "source_id", "ra", "dec", "parallax", "parallax_error",',
    '        "pmra", "pmdec", "phot_g_mean_mag", "phot_bp_mean_mag",',
    '        "phot_rp_mean_mag", "bp_rp", "g_rp", "ruwe",',
    '        "ebpminrp_gspphot", "bp_rp_0"',
    '    ]',
    '    retry_lookup = retry_gaia.set_index("source_id")',
    '',
    '    mask = (',
    '        df_final["source_id"].isna() &',
    '        df_final["gaia_dr3_source_id_from_simbad"].notna()',
    '    )',
    '    for idx, row in df_final[mask].iterrows():',
    '        sid = str(row["gaia_dr3_source_id_from_simbad"])',
    '        if sid in retry_lookup.index:',
    '            gr = retry_lookup.loc[sid]',
    '            if isinstance(gr, pd.DataFrame):',
    '                gr = gr.iloc[0]',
    '            for col in gaia_fill_cols:',
    '                if col in gr.index:',
    '                    df_final.at[idx, col] = gr[col]',
    '',
    'print(f"After direct retry fill:")',
    'print(f"  With Gaia source_id: {df_final[\'source_id\'].notna().sum()}")',
    'print(f"  Still missing:       {df_final[\'source_id\'].isna().sum()}")',
])

# ── New cell 77c: cone search for stars with no Gaia ID at all ──────────────
cell77c_source = '\n'.join([
    '# Step 2: cone search for stars that have SIMBAD coords but no Gaia ID.',
    '# (These never had a source_id to query directly.)',
    'cone_results  = []',
    'cone_failures = []',
    '',
    'for i, row in enumerate(needs_cone.itertuples(), start=1):',
    '    print(f"{i}/{len(needs_cone)}: {row.star_name}")',
    '    try:',
    '        candidates = gaia_cone_search(row.simbad_ra_deg, row.simbad_dec_deg, radius_arcsec=5)',
    '        best, flag = choose_best_gaia_match(candidates)',
    '        if best is not None:',
    '            best["simbad_ra_deg"] = row.simbad_ra_deg',
    '            best["simbad_dec_deg"] = row.simbad_dec_deg',
    '            best["cone_flag"] = flag',
    '            cone_results.append(best)',
    '            print(f"  -> matched source_id={best[\'source_id\']}, dist={best[\'dist_arcsec\']:.2f}\'")',
    '        else:',
    '            cone_failures.append({"star_name": row.star_name, "reason": flag})',
    '            print(f"  -> no match ({flag})")',
    '        time.sleep(0.5)',
    '    except Exception as e:',
    '        cone_failures.append({"star_name": row.star_name, "reason": str(e)})',
    '        print(f"  -> ERROR: {e}")',
    '        time.sleep(2.0)',
    '',
    'print(f"\\nCone search matched: {len(cone_results)}")',
    'print(f"Still unmatched:     {len(cone_failures)}")',
])

# Find indices of cells 76 and 77
idx76 = next(i for i, c in enumerate(nb['cells']) if c['id'] == '7fecb96d')
idx77 = next(i for i, c in enumerate(nb['cells']) if c['id'] == '4348e058')

# Replace cell 76
nb['cells'][idx76]['source'] = cell76_source

# Replace cell 77
nb['cells'][idx77]['source'] = cell77_source

# Insert cell 77b and 77c after idx77
nb['cells'].insert(idx77 + 1, make_code_cell(cell77b_source))
nb['cells'].insert(idx77 + 2, make_code_cell(cell77c_source))

# Also update cell 78 (inspect cone results) — it now comes after 77c
# Its content is fine as-is, no change needed.

with open('engle_xmatch.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Done. Notebook now has {len(nb["cells"])} cells.')
# Print cells 76-83 to verify
for i in range(idx76, idx76 + 8):
    c = nb['cells'][i]
    print(f'  Cell {i} ({c["id"]}): {chr(10).join(c["source"].splitlines()[:2])}')
