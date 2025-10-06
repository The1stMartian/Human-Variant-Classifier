#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Run in docker container variant_model:v1
# - has all python libraries (pandas, scikitlearn)
# - has VEP on path and dictionary
# - has human genome .fa file
# - has bcf tools

# Inputs:
#    - parquet file of known variants
#    - query file of CHROM, POS, REF, ALT
#    - gene-level constraint data file with 1 row per gene (no dups for alt splicing)
#    - human genome fasta sequence file
#    - .vep folder with the human genome data in it (in docker container)
#    - small model file

# Outputs:
#    - small model query data in .tsv format
#    - small model query data + prediction .tsv
#    - large model query data + prediction if available
#    - vep results .tsv

import os, sys, tempfile, subprocess, re
import pandas as pd
import joblib
import shutil

# -------------------- Configuration (no argparse) --------------------
INPUT_TSV         = "/input/variants.tsv"                  # columns: CHROM, POS, REF, ALT
GENE_METRICS_TSV  = "/input/geneLevelDataFiltered.tsv"     # columns: gene, loeuf, pli, mis_z, syn_z
REF_FASTA         = "/input/grch38.fa"                     # and .fai alongside it
KNOWN_VARS        = "/input/allDataWithPreds.parquet"      # Known variant data + large model effect predictions
MODEL_PATH        = "/input/smallModel.joblib"             # Small model location
VEP_CACHE_DIR     = "~/.vep"                               # mount VEP library
CHROM_PREFIX      = "chr"
OUT_FOLDER        = sys.argv[5]                             # should be "/input/output" 
OUTPUT_TSV        = OUT_FOLDER + "/" + "smallModelQuery.tsv"
OUTPUT_KNOWN_VAR  = OUT_FOLDER + "/" + sys.argv[6]
OUTPUT_MODEL_INFO = OUT_FOLDER + "/" + sys.argv[7]
OUTPUT_VEP        = OUT_FOLDER + "/" + sys.argv[8]

# VEP command (no -i/-o here; added by the script)
VEP_CACHE_DIR = os.path.expanduser(VEP_CACHE_DIR) # expand the path if using "~/"

VEP_BASE_CMD = (
    "vep --cache --offline --assembly GRCh38 "
    "--format vcf --vcf --symbol --everything "
    f"--dir_cache {VEP_CACHE_DIR} --fasta {REF_FASTA}"
)

# SpliceAI command (standard CLI)
SPLICEAI_CMD = "spliceai"
SPLICEAI_ARGS = "-D 4999 -M 1 -A grch38 "  # ±5kb, masking on, GRCh38 annotation

# -------------------- Helpers --------------------
PURINES = {"A", "G"}
PYRIMIDINES = {"C", "T"}
MITO_CHROMS = {"MT", "M", "chrM", "chrMT"}

def queryKnownVars(library=KNOWN_VARS,chr=None,pos=None,ref=None,alt=None, knownVarOutPath=None):
    '''
    - Query the parquet formatted dataframe for the variant
    - Returns: none
    - Output: A .tsv file with results or nothing
    '''
    print("Querying known variant data...\n")
    pos = int(pos)
    p = pd.read_parquet(library)
  
    # Ensure chromosome is "chr1" format, not just a number
    if "chr" not in chr:
        chr = "chr" + chr
    variantData = p[(p["chrom"] == chr) & (p["pos"] == pos) & (p["ref"] == ref) & (p["alt"] == alt)]
    
    print()
    print("VARDATA: ", variantData)
    print()
    
    # Unknown variant > do nothing
    if len(variantData) >= 1:
        variantData.to_csv(knownVarOutPath, index=False, sep="\t")
    
    return None

def run(cmd, timeout=1200):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nSTDERR:\n{p.stderr.decode()}")
    return p.stdout.decode()

def write_batch_vcf(df, path, chrom_prefix=None):
    with open(path, "w") as fh:
        for _, r in df.iterrows():
            chrom = str(r["chrom"])
            if chrom_prefix and not chrom.startswith(chrom_prefix):
                chrom = chrom_prefix + chrom.lstrip("chr")
            vcfLine = f"{chrom}\t{int(r['pos'])}\t.\t{r['ref']}\t{r['alt']}\t.\t.\t.\n" 
        
        # Write header
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=" + chrom + ">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        # Write variant
        fh.write(vcfLine)

def compute_basics(ref, alt):
    rl = len(ref); al = len(alt)
    is_snv = int(rl == 1 and al == 1)
    is_indel = int(not is_snv)
    indel_len = abs(rl - al) if is_indel else 0
    is_transition_num = int((ref in PURINES and alt in PURINES) or (ref in PYRIMIDINES and alt in PYRIMIDINES)) if is_snv else 0
    # heuristic frameshift; VEP can override if frameshift consequence present
    is_frameshift = int(is_indel == 1 and (indel_len % 3 != 0))
    return dict(ref_len=rl, alt_len=al, is_indel=is_indel, is_snv=is_snv,
                indel_len=indel_len, is_transition_num=is_transition_num,
                is_frameshift=is_frameshift)

def parse_spliceai_info(info):
    # SpliceAI=SYMBOL|TX|DS_AG|DS_AL|DS_DG|DS_DL|...
    out = {"spliceai_tx_count": None, "spliceai_ds_ag_max": None, "spliceai_ds_al_max": None,
           "spliceai_ds_dg_max": None, "spliceai_ds_dl_max": None}
    if not info or info == ".":
        return out
    tx=0; ag=al=dg=dl=None
    for entry in info.split(","):
        parts = entry.split("|")
        if len(parts) >= 6:
            tx += 1
            def f(x):
                try: return float(x)
                except: return None
            _ag,_al,_dg,_dl = map(f, parts[2:6])
            ag = _ag if (ag is None or (_ag is not None and _ag > ag)) else ag
            al = _al if (al is None or (_al is not None and _al > al)) else al
            dg = _dg if (dg is None or (_dg is not None and _dg > dg)) else dg
            dl = _dl if (dl is None or (_dl is not None and _dl > dl)) else dl
    out.update({"spliceai_tx_count": (tx or None), "spliceai_ds_ag_max": ag,
                "spliceai_ds_al_max": al, "spliceai_ds_dg_max": dg, "spliceai_ds_dl_max": dl})
    return out

def parse_vep(vcf_path):
    # Returns dict[(chrom,pos,ref,alt)] -> flags + symbol
    csq_headers = None
    res = {}
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("##INFO=<ID=CSQ"):
                i = line.find("Format: ")
                if i != -1:
                    fmt = line[i+8:].split('">')[0].strip()
                    csq_headers = fmt.split("|")
            
            # Get info types for VEP
            if line.startswith("##INFO="):
                vepInfotype = []
                x = line.strip().replace("##INFO=<", "")
                y = x.split(",")
                for z in y:
                    if z.lower().startswith("desc"):
                        z = z.split(":")[1]
                        a = z.split("|")
                        for infoField in a:
                            vepInfotype.append(infoField)

            if line.startswith("#"): continue
            cols = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = cols[0], int(cols[1]), cols[3], cols[4]
            info = cols[7]
            csq = None
            for tok in info.split(";"):
                if tok.startswith("CSQ="): csq = tok[4:]; break
            vals = {"is_missense":0, "is_frameshift":0, "symbol":None,
                    "mt_missense":0, "mt_noncoding":0, "mt_nonsense":0, "mt_silent":0}
            if csq and csq_headers:
                
                # Collect VEP output data to go with headers
                vepResults = []
                for item in csq.split(",")[0].split("|"):
                    for p in item.split("|"):
                        vepResults.append(p)

                for item in csq.split(","):
                    parts = item.split("|")
                    d = {csq_headers[i]: (parts[i] if i < len(csq_headers) else "") for i in range(len(csq_headers))}
                    cons = d.get("Consequence","")
                    sym  = d.get("SYMBOL") or None
                    if sym and vals["symbol"] is None: vals["symbol"] = sym
                    if "missense_variant" in cons: vals["is_missense"] = 1
                    if "frameshift_variant" in cons: vals["is_frameshift"] = 1
                    if chrom in MITO_CHROMS:
                        if "missense_variant" in cons: vals["mt_missense"] = 1
                        if "stop_gained" in cons: vals["mt_nonsense"] = 1
                        if "synonymous_variant" in cons: vals["mt_silent"] = 1
                        if ("non_coding" in cons) or ("intron_variant" in cons) or ("intergenic_variant" in cons) or ("UTR" in cons):
                            vals["mt_noncoding"] = 1
            res[(chrom, pos, ref, alt)] = vals
    return res, vepInfotype, vepResults

def load_gene_metrics(path):
    # expected columns: gene, loeuf, pli, mis_z, syn_z
    df = pd.read_csv(path, sep="\t", dtype=str)
    def to_float(x):
        try: return float(x)
        except: return None
    gm = {}
    for _, r in df.iterrows():
        gene = str(r["gene"])
        gm[gene] = {"loeuf": to_float(r.get("loeuf")),
                    "pli":   to_float(r.get("pli")),
                    "mis_z": to_float(r.get("mis_z")),
                    "syn_z": to_float(r.get("syn_z"))}
    return gm

def analyze(predValue):
    '''
    - Converts model output score to semantic value
    - Input: prediction value
    - Ouput: "benign", "pathogenic", Indeterminate
    '''
    if predValue >= 0.75: return "Likely Pathogenic"
    elif predValue <= 0.25: return "Likely Benign"
    else: return "Indeterminate"

def predict(input_tsv_file, model_preds_outfile_path, modelPath):
    '''
    - The output .tsv query file has the full small model query
    - Input: OUTPUT_TSV file
    - Output: pandas df with one row
        - query line data
        - prediction score
        - prediction (likely benign/pathogenic)
    '''
    
    # Load the model
    smallModel = joblib.load(modelPath)
    
    # Load the .tsv file0
    query = pd.read_csv(input_tsv_file, sep="\t")
    
    # Define query data - remove unwanted columns and index
    X = query.drop(["CHROM","POS","REF","ALT", "mpc_filled", "mpc_is_missing"], axis =1 )
    X = X.reset_index(drop=True)

    # Predict outcome of mutation
    proba = smallModel.predict_proba(X)[:,1]
    
    # Append probability and prediction 
    query["probability"] = proba
    query["prediction"] = analyze(proba)

    # Write outfile (only for testing)
    query.to_csv(model_preds_outfile_path, index=False, sep="\t")

    # Return the full small model input & model prediction
    return None

def saveVepToFile(vepHeaders, vepRes, outPath):
    '''
    - Saves all VEP out data to out file
    - Inputs: (2), list:header, list:1 row of results
    - Output: (1), csv file
    '''
    vepRes = [vepRes]
    df = pd.DataFrame(vepRes, columns=vepHeaders)
    df.to_csv(outPath, sep="\t", index=False)

# -------------------- Main (no argparse) --------------------
def main():

    # Load query (expects CHROM, POS, REF, ALT) as arguments
    print("Loading query data...")
    if len(sys.argv) != 9:
        print()
        print("Error: Expected usage 'python queryAndPredict.py chrom pos ref alt outputFolderPath'")
        print("Try running with a full query.")
        print()
        exit()
    queryChr  = str(sys.argv[1])
    queryPos = int(sys.argv[2])
    queryRef = str(sys.argv[3])
    queryAlt = str(sys.argv[4])
    
    dataList = [[queryChr, queryPos, queryRef, queryAlt]]
    qv = pd.DataFrame(dataList, columns = ["chrom", "pos", "ref", "alt"])

    # Preflight
    for p in [GENE_METRICS_TSV, REF_FASTA]:
        if not os.path.exists(p):
            sys.exit(f"Missing required file: {p}")
    if not os.path.isdir(VEP_CACHE_DIR):
        sys.exit(f"VEP cache directory not found: {VEP_CACHE_DIR}")

    print("All input files found. Continuing...")

    # Make out directory
    # Cleanup previous outputs:
    if os.path.exists(OUT_FOLDER):
        shutil.rmtree(OUT_FOLDER)
    print("EXISTS? ", os.path.exists(OUT_FOLDER))
    os.makedirs(OUT_FOLDER, exist_ok=False)

    # Look up query in known variant database
    queryKnownVars(chr=queryChr, pos=queryPos, ref=queryRef, 
        alt=queryAlt, knownVarOutPath=OUTPUT_KNOWN_VAR)

    # Compute basics
    basics = []
    for _, r in qv.iterrows():
        basics.append(compute_basics(r["ref"], r["alt"]))
    basics_df = pd.DataFrame(basics, index=qv.index)

    with tempfile.TemporaryDirectory() as td:
        print("Running VEP...")
        vcf_in = os.path.join(td, "variants.vcf")
        write_batch_vcf(qv, vcf_in, chrom_prefix=CHROM_PREFIX)

        # VEP
        vep_out = os.path.join(td, "vep_out.vcf")
        # ensure no stray -i/-o in base cmd
        base_cmd = re.sub(r"(?:^|\s)-(?:i|o)\s+\S+|(?:^|\s)--(?:input_file|output_file)\s+\S+", "", VEP_BASE_CMD)
        base_cmd = re.sub(r"\s+", " ", base_cmd).strip()
        print(f"VEP COMMAND: {base_cmd} -i {vcf_in} -o {vep_out} \n")
        run(f"{base_cmd} -i {vcf_in} -o {vep_out}")
        vep_map, vepHeaders, vepRes = parse_vep(vep_out)

        # SpliceAI
        print("Running SpliceAI...")
        sp_out = os.path.join(td, "spliceai_out.vcf")
        print(f"SpliceAI command: {SPLICEAI_CMD} -I {vcf_in} -O {sp_out} -R {REF_FASTA} {SPLICEAI_ARGS}\n")
        run(f"{SPLICEAI_CMD} -I {vcf_in} -O {sp_out} -R {REF_FASTA} {SPLICEAI_ARGS}")
        splice_map = {}
        with open(sp_out) as fh:
            for line in fh:
                if line.startswith("#"): continue
                c = line.rstrip("\n").split("\t")
                key = (c[0], int(c[1]), c[3], c[4])
                info = c[7]
                sv = next((t.split("=",1)[1] for t in info.split(";") if t.startswith("SpliceAI=")), None)
                splice_map[key] = parse_spliceai_info(sv)

    # Load gene metrics
    gene_metrics = load_gene_metrics(GENE_METRICS_TSV)

    # Assemble output records
    fields = [
        "loeuf","pli","mis_z","syn_z",
        "ref_len","alt_len","is_indel","is_snv","indel_len","is_transition_num",
        "spliceai_tx_count","spliceai_ds_ag_max","spliceai_ds_al_max","spliceai_ds_dg_max","spliceai_ds_dl_max",
        "is_frameshift","is_missense",
        "mpc_filled","mpc_is_missing",
        "mt_missense","mt_noncoding","mt_nonsense","mt_silent",
    ]

    recs = []
    for i, r in qv.iterrows():
        # Key used by VEP/SpliceAI (with chr prefix normalization)
        chrom_key = str(r["chrom"])
        if CHROM_PREFIX and not chrom_key.startswith(CHROM_PREFIX):
            chrom_key = CHROM_PREFIX + chrom_key.lstrip("chr")
        key = (chrom_key, int(r["pos"]), r["ref"], r["alt"])

        rec = {f: None for f in fields}
        rec.update(basics_df.loc[i].to_dict())

        # Default MPC flags (not computing MPC)
        rec["mpc_filled"] = 0
        rec["mpc_is_missing"] = 1

        # Merge VEP
        vm = vep_map.get(key, {})
        if vm:
            rec["is_missense"] = vm.get("is_missense", 0)
            if vm.get("is_frameshift", 0) == 1:
                rec["is_frameshift"] = 1
            rec["mt_missense"]  = vm.get("mt_missense", 0)
            rec["mt_noncoding"] = vm.get("mt_noncoding", 0)
            rec["mt_nonsense"]  = vm.get("mt_nonsense", 0)
            rec["mt_silent"]    = vm.get("mt_silent", 0)
            sym = vm.get("symbol")
            if sym and sym in gene_metrics:
                rec.update({k: gene_metrics[sym][k] for k in ("loeuf","pli","mis_z","syn_z")})

        # Merge SpliceAI
        rec.update(splice_map.get(key, {}))

        recs.append(rec)

    out = pd.DataFrame(recs)
    out.insert(0, "CHROM", qv["chrom"].values)
    out.insert(1, "POS",   qv["pos"].values)
    out.insert(2, "REF",   qv["ref"].values)
    out.insert(3, "ALT",   qv["alt"].values)

    # Write TSV
    out.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"Wrote {OUTPUT_TSV} with {len(out)} variants.")


    # Make prediction - saves to out file "/input/modelOut.csv"
    print("Using the small model to make predictions...")
    predict(OUTPUT_TSV, OUTPUT_MODEL_INFO, MODEL_PATH)

    # Save vep predictions to "/input/vepOut.csv"
    print("Saving VEP data to an outfile...")
    saveVepToFile(vepHeaders, vepRes, OUTPUT_VEP)

    print("Done!")
    print()

if __name__ == "__main__":
    main()
