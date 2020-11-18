#!/usr/bin/python3
"""
    SQUEzE - WEAVE
    ==============
    Python wrapper to run SQUEzE code on WEAVE DATA and generate output tables.
    Results from SQUEzE are used as priors in a run of RedRock.

    versions:
     1.0 By Ignasi Perez-Rafols (LPNHE, July 2020) - First version


########### BAISC APS PARAM ###########################################################################################################
|    param                |    APS default  |  RRWEAVE EQUIVALENT                 |  RRWEAVE DEFAULT  |   availble options/notes      |
infiles (Required)        |        -        |  --infiles          (Required)      |        -          |                               |
aps_ids (Optional)        |     None        |  --aps_ids          (Optional)      |        None       |                               |
targsrvy (Optional)       |     None        |  --targsrvy         (Optional)      |        None       |                               |
targclass (Optional)      |     None        |  --targclass        (Optional)      |        None       |                               |
mask_aps_ids (Optional)   |     None        |  --mask_aps_ids     (Optional)      |        None       |                               |
wlranges (Optional)       |     None        |  --wlranges         (Optional)      |        None       |                               |
area (Optional)           |     None        |  --area             (Optional)      |        None       |                               |
mask_areas (Optional)     |     None        |  --mask_areas       (Optional)      |        None       |                               |
sens_corr (Optional)      |     True        |  --sens_corr        (Optional)      |        True       |                               |
mask_gaps (Optional)      |     True        |  --mask_gaps        (Optional)      |        True       |                               |
vacuum (Optional)         |     False       |  --vacuum           (Optional)      |        True       |                               |
tellurics (Optional)      |     False       |  --tellurics        (Optional)      |        False      |                               |
fill_gap (Optional)       |     False       |  --fill_gap         (Optional)      |        False      |                               |
arms_ratio (Optional)     |     None        |  --arms_ratio       (Optional)      |        None       | for R band in OPR3B  is 0.83  |
join_arms (Optional)      |     False       |  --join_arms        (Optional)      |        False      |                               |
funit (Optional)          |     1.0e18      |  USE DEFAULT APS value              |        AS APS     |                               |
offset_gap_pix (Optional) |     10          |  USE DEFAULT APS value              |        AS APS     |                               |
                          |                 |                                     |                   |                               |
######### RRWEAVE INITAIL PARAM #######################################################################################################
cache_Rcsr  (Optional)    |     -           |  --cache_Rcsr       (Optional)      |         True      |                               |
                          |                 |                                     |                   |                               |
######### DEDICATED RRWEAVE PARAM #####################################################################################################
                          |                 |  --outpath          (Required)      |          -        |                               |
                          |                 |  --srvyconf         (Required)      |          -        |                               |
                          |                 |  --templates        (Optional)      |        None       |                               |
                          |                 |  --headname         (Optional)      |     'headname'    |                               |
                          |                 |  --fig              (Optional)      |        False      |                               |
                          |                 |  --overwrite        (Optional)      |        False      |                               |
                          |                 |  --mp               (Optional)      |         2         |                               |
                          |                 |  --archetypes       (Optional)      |        None       |                               |
                          |                 |  --zall             (Optional)      |        False      |                               |
                          |                 |  --chi2_scan        (Optional)      |        None       |                               |
                          |                 |  --debug            (Optional)      |        False      |                               |
                          |                 |  --nminima          (Optional)      |        3          |                               |
                          |                 |                                     |                   |                               |
######### DEDICATED SQUEZE_WEAVE PARAM ################################################################################################
                          |                 |  --model            (Required)      |          -        |                               |
                          |                 |  --model_fits       (Optional)      |        False      |                               |
                          |                 |  --prob_cut         (Optional)      |          -        |                               |
                          |                 |  --output_catalogue (Required)      |        0.0        |                               |                   |                               |
                          |                 |  --quiet            (Optional)      |        False      |
                          |                 |  --clean_dir        (Optional)      |        True       |                               |
                          |                 |                                     |                   |                               |
#######################################################################################################################################



Example:

"""
__author__ = "Ignasi Perez-Rafols (iprafols@gmail.com)"
__version__ = "0.1"

import argparse
import os
import sys
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.table import Table, Column, hstack, join
from collections import OrderedDict
import json
from datetime import datetime

from aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist
from aps_rr import write_ztable, gen_zfitall, gen_zbest, gen_zspec, rrweave_worker
import aps_constants
import warnings

from redrock.utils import elapsed, get_mp, distribute_work
from redrock.targets import Spectrum, Target, DistTargetsCopy
from redrock.templates import load_dist_templates
from redrock.results import write_zscan
from redrock.zfind import zfind
from redrock._version import __version__
from redrock.archetypes import All_archetypes

from squeze.common_functions import verboseprint, quietprint
from squeze.weave_spectrum import WeaveSpectrum
from squeze.common_functions import load_json
from squeze.model import Model
from squeze.spectra import Spectra
from squeze.candidates import Candidates
from squeze.parsers import PARENT_PARSER, QUASAR_CATALOGUE_PARSER

def squeze_worker(infiles, model, aps_ids, targsrvy, targclass, mask_aps_ids,
                  area, mask_areas, wlranges, cache_Rcsr, sens_corr, mask_gaps,
                  vacuum, tellurics, fill_gap, arms_ratio, join_arms,
                  quiet=False, save_file=None):

    """
    Function description:
        Run SQUEzE on the data from infiles
    """

    # manage verbosity
    userprint = verboseprint if not quiet else quietprint

    # load model
    userprint("================================================")
    userprint("")
    userprint("Loading model")
    if model.endswith(".json"):
        model = Model.from_json(load_json(model))
    else:
        model = Model.from_fits(model)

    # load spectra
    userprint("Loading spectra")
    weave_formatted_spectra = APSOB(infiles, aps_ids=aps_ids,
                                    targsrvy= targsrvy, targclass = targclass,
                                    mask_aps_ids = mask_aps_ids, area=area,
                                    mask_areas=mask_areas, wlranges=wlranges,
                                    sens_corr=sens_corr, mask_gaps=mask_gaps,
                                    vacuum=vacuum, tellurics=tellurics,
                                    fill_gap=fill_gap, arms_ratio=arms_ratio,
                                    join_arms=join_arms)

    userprint("Formatting spectra to be digested by SQUEzE")
    spectra = Spectra.from_weave(weave_formatted_spectra, userprint=userprint)

    # TODO: split spectra into several sublists so that we can parallelise

    # initialize candidates object
    userprint("Initialize candidates object")
    if save_file is None:
        candidates = Candidates(mode="operation", model=model)
    else:
        candidates = Candidates(mode="operation", model=model, name=save_file)

    # look for candidates
    userprint("Looking for candidates")
    if save_file is None:
        candidates.find_candidates(spectra.spectra_list(), save=False)
    else:
        candidates.find_candidates(spectra.spectra_list(), save=True)

    # compute probabilities
    userprint("Computing probabilities")
    candidates.classify_candidates()

    # TODO: if we paralelize, then use 'merging' mode to join the results

    userprint("SQUEzE run is completed, returning")
    userprint("================================================")
    userprint("")
    # return the candidates and the chosen probability threshold
    return candidates.candidates(), model.get_settings().get("Z_PRECISION")

def write_results(zbest, candidates_df, args):
    """ Format results according to the specifications of the CS
    specifications

    Args:
        zfitall (astropy.TABLE): Table with the redrock results run with SQUEzE priors
        candidates_df (pd.DataFrame): DataFrame with SQUEzE candidates' information
        args: Parsed arguments from main
    """
    # first create the primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["COMMENT"] = "WEAVE Contributed Software: SQUEzE"
    primary_hdu.header["DATAMVER"] = ("8.00", "WEAVE Data Model Version")
    # TODO: read the version of the CS on the fly
    primary_hdu.header["CSVER"] = ("0.4", "CS version")
    primary_hdu.header["CS_NME1"] = ("Ignasi", "CS author forename")
    primary_hdu.header["CS_NME1"] = ("Perez Rafols", "CS author surname(s)")
    primary_hdu.header["CS_MAIL"] = ("iprafols@gmail.com", "CS author email")
    # TODO: update the files used
    primary_hdu.header["PROV0001"] = ("", "L1 file used")
    primary_hdu.header["PROV0002"] = ("", "L2 file used")
    now = datetime.now()
    primary_hdu.header["DATETIME"] = (now.strftime("%Y-%m-%dT%H:%M:%S"), "DateTime file created")
    # TODO: update the datasum
    primary_hdu.header["DATASUM"] = ("", "Generated by the SPA")

    # remove duplicates
    filtered_candidates = candidates_df[(~candidates_df["DUPLICATED"])]

    # sanity check: apply probability cut
    filtered_candidates = filtered_candidates[(filtered_candidates["PROB"] >= args.prob_cut)]

    # then create the final catalogue
    aps_ids = filtered_candidates["APS_ID"]
    cnames = filtered_candidates["CNAME"]
    targids = filtered_candidates["TARGID"]
    targclasses = ["QSO"]*cnames.size
    probs = filtered_candidates["PROB"]
    z = [zbest[zbest["APS_ID"] == aps_id]["Z"][0] for aps_id in aps_ids]
    cols  = [
        fits.Column(name="CNAME",
                    format="20A",
                    disp="A20",
                    array=cnames,
                   ),
        fits.Column(name="TARGID",
                    format="30A",
                    disp="A30",
                    array=targids,
                   ),
        fits.Column(name="TARGCLASS",
                    format="12A",
                    disp="A30",
                    array=targclasses,
                   ),
        fits.Column(name="SQUEZE_PROB",
                    format="E",
                    disp="F7.3",
                    array=probs,
                   ),
        fits.Column(name="SQUEZE_Z",
                    format="E",
                    disp="F7.3",
                    array=z,
                   ),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="QUASAR_IDS")

    hdu.header["TUCD1"] = "meta.id;meta.main"
    hdu.header["TUCD2"] = "meta.id"
    hdu.header["TUCD3"] = "src.class"
    hdu.header["TUCD4"] = "stats.probability"
    hdu.header["TUCD5"] = "src.redshift"
    hdu.header["TPROP1"] = 0
    hdu.header["TPROP2"] = 0
    hdu.header["TPROP3"] = 0
    hdu.header["TPROP4"] = 0
    hdu.header["TPROP5"] = 0
    hdu.header["TDMIN4"] = 0.0
    hdu.header["TDMIN5"] = 0.0
    hdu.header["TDMAX4"] = 1.0
    hdu.header["DATASUM"] = ""

    desc = {"XTENSION": "binary table extension",
            "BITPIX": "array data type",
            "NAXIS": "number of array dimensions",
            "NAXIS1": "length of dimension 1",
            "NAXIS2": "length of dimension 2",
            "PCOUNT": "number of group parameters",
            "GCOUNT": "number of groups",
            "TFIELDS": "number of table fields",
            "EXTNAME": "CS code SQUEzE extension name",
            "TTYPE1": "WEAVE object name from coordinates",
            "TFORM1": "data format of field: ASCII Character",
            "TDISP1": "Display format for column",
            "TUCD1": "UCD for column",
            "TPROP1": "Public column",
            "TTYPE2": "The identifier of the target assigned by survey",
            "TFORM2": "data format of field: ASCII Character",
            "TDISP2": "Display format for column",
            "TUCD2": "UCD for column",
            "TPROP2": "Public column",
            "TTYPE3": "Classification of the target assigned by survey",
            "TFORM3": "data format of field: ASCII Character",
            "TDISP3": "Display format for column",
            "TUCD3": "UCD for column",
            "TPROP3": "Public column",
            "TTYPE4": "Confidence",
            "TFORM4": "data format of field: 4-byte REAL",
            "TDISP4": "Display format for column",
            "TUCD4": "UCD for column",
            "TDMIN4": "--",
            "TDMAX4": "--",
            "TPROP4": "Public column",
            "TTYPE5": "SQUEzE redshift",
            "TFORM5": "data format of field: 4-byte REAL",
            "TDISP5": "Display format for column",
            "TUCD5": "UCD for column",
            "TDMIN5": "--",
            "TPROP5": "Public column",
            "DATASUM": "Generated by the SPA",
            }
    for key in hdu.header:
        hdu.header.comments[key] = desc.get(key, "")

    # now create the HDU with all SQUEzE classifications
    num_peaks = 15
    cnames = filtered_candidates["CNAME"]
    probs = np.zeros(num_peaks)
    z = np.zeros(num_peaks)
    for cname in cnames:
        candidates_per_cname = candidates_df[candidates_df["CNAME"] == cname]
        for index in range(num_peaks):
            try:
                probs[index] = candidates_per_cname["PROB"].iloc[index]
                z[index] = candidates_per_cname["Z_TRY"].iloc[index]
            except IndexError:
                probs[index] = np.nan
                z[index] = np.nan
    cols = [
        fits.Column(name="CNAME",
                    format="20A",
                    disp="A20",
                    array=cnames,
                   ),
        fits.Column(name="SQUEZE_PROB",
                    format="{}E".format(num_peaks),
                    disp="F7.3",
                    array=probs,
                   ),
        fits.Column(name="SQUEZE_Z",
                    format="{}E".format(num_peaks),
                    disp="F7.3",
                    array=z,
                   ),
    ]
    hdu2 = fits.BinTableHDU.from_columns(cols, name="QUASAR_ALT_CLASS")

    hdu2.header["TUCD1"] = "meta.id;meta.main"
    hdu2.header["TUCD4"] = "stats.probability"
    hdu2.header["TUCD5"] = "src.redshift"
    hdu2.header["TPROP1"] = 0
    hdu2.header["TPROP4"] = 0
    hdu2.header["TPROP5"] = 0
    hdu2.header["TDMIN4"] = 0.0
    hdu2.header["TDMIN5"] = 0.0
    hdu2.header["TDMAX4"] = 1.0
    hdu2.header["DATASUM"] = ""

    desc = {"XTENSION": "binary table extension",
            "BITPIX": "array data type",
            "NAXIS": "number of array dimensions",
            "NAXIS1": "length of dimension 1",
            "NAXIS2": "length of dimension 2",
            "PCOUNT": "number of group parameters",
            "GCOUNT": "number of groups",
            "TFIELDS": "number of table fields",
            "EXTNAME": "CS code SQUEzE extension name",
            "TTYPE1": "WEAVE object name from coordinates",
            "TFORM1": "data format of field: ASCII Character",
            "TDISP1": "Display format for column",
            "TUCD1": "UCD for column",
            "TPROP1": "Public column",
            "TTYPE2": "Alternative Confidences",
            "TFORM2": "data format of field: 4-byte REAL",
            "TDISP2": "Display format for column",
            "TUCD2": "UCD for column",
            "TDMIN2": "--",
            "TDMAX2": "--",
            "TPROP2": "Public column",
            "TTYPE3": "Alternative SQUEzE redshift ",
            "TFORM3": "data format of field: 4-byte REAL",
            "TDISP3": "Display format for column",
            "TUCD3": "UCD for column",
            "TDMIN3": "--",
            "TPROP3": "Public column",
            "DATASUM": "Generated by the SPA",
            }
    for key in hdu2.header:
        hdu2.header.comments[key] = desc.get(key, "")

    # finally save the catalogue
    if args.output_catalogue.startswith("/"):
        filename = args.output_catalogue
    else:
        filename = args.output_path + args.output_catalogue
    hdul = fits.HDUList([primary_hdu, hdu, hdu2])
    hdul.writeto(filename,
                 overwrite=True, checksum=True)

def main(options=None, comm=None):
    """ Run SQUEzE on WEAVE data

    This loads targets serially and runs SQUEzE on them, producing rought
    redshift estimates. It then runs RedRock for fine-tuning of the redshift
    fitting and writes the output to a catalog.

    Args:
        options (list): optional list of commandline options to parse.
        comm (mpi4py.Comm): MPI communicator to use.
    """
    parser = argparse.ArgumentParser(description="Estimate redshifts from"
        " WEAVE target spectra.")

    parser.add_argument("--infiles", type=none_or_str, default=None,
        required=True, help="input files", nargs='*')

    parser.add_argument("--aps_ids", type=none_or_str, default=None,
        required=False, help="comma-separated list of WEAVE APS_IDs to be considered")

    parser.add_argument("--targsrvy", type=none_or_str, default=None,
        required=False, help="comma-separated list of surveys to be considered")

    parser.add_argument("--targclass", type=none_or_str, default=None,
        required=False, help="comma-separated list of classtypes to be considered")

    parser.add_argument("--mask_aps_ids", type=none_or_str, default=None,
        required=False, help="comma-separated list of APS_IDs to be masked")

    parser.add_argument("--wlranges", type=none_or_str, default=None,
        required=False, help="wavelength range array for each elements of the setup", nargs='*')

    parser.add_argument("--area", type=none_or_str, default=None,
        required=False, help="The area in [RA(deg), DEC(deg), Radius(arcsec)] to be considered in analysis")

    parser.add_argument("--mask_areas", type=none_or_str, default=None,
        required=False,
        help="The multi area(s) in [RA(deg), DEC(deg), Radius(arcsec)] to be excluded from analysis", nargs='*')

    parser.add_argument("--sens_corr", type=str2bool, default=True,
        required=False, help="apply sensitivity function to the flux and ivar")

    parser.add_argument("--mask_gaps", type=str2bool, default=True,
        required=False, help="mask the gaps between CCD by putting ivar = 0")

    parser.add_argument("--vacuum", type=str2bool, default=True,
        required=False, help="transform wavelenght from air to vacuum")

    parser.add_argument("--tellurics", type=str2bool, default=False,
        required=False, help="put large errors for regions affected by tellurics")

    parser.add_argument("--fill_gap", type=str2bool, default=False,
        required=False, help="fill CCD gaps by interpolated values for flux")

    parser.add_argument("--arms_ratio", type=none_or_str, default=None,
        required=False, help="Correction (flux and ivar/error) fraction for each arms")

    parser.add_argument("--join_arms", type=str2bool, default=False,
        required=False, help="Stitch two arms")

    parser.add_argument('--outpath',
        help='Directory to keep WEAVE_REDROCK outputs', type=str, default=None, required=True)

    parser.add_argument("--srvyconf", type=none_or_str, default=None,
        required=True, help="json config file (Pre-defined class type, based on TARGSRVY)")

    parser.add_argument("-t", "--templates", type=none_or_str, default=None,
        required=False, help="template file or directory")

    parser.add_argument('--headname',
        help='Output headname. The output filenames will be generated based on this',
        type=str, default='headname', required=True)

    parser.add_argument("--fig", default=False, type=str2bool,
        required=False, help="if True, the code also produces plots of the best fitted model")

    parser.add_argument('--overwrite',
                        help='If True, overwrites the existing products, otherwise it will skip them',
                        type=str2bool,  default=False)

    parser.add_argument("--mp", type=int, default=1,
        required=False, help="if not using MPI, the number of multiprocessing"
            " processes to use (defaults to half of the hardware threads)")

    parser.add_argument("--archetypes", type=none_or_str, default=None,
        required=False, help="archetype file or directory for final redshift comparisons")

    parser.add_argument("--zall", type=str2bool, default=False,
        required=False, help="if True, it creates a fits file contains all WEAVE_REDROCK outputs. [FITS file]")

    #parser.add_argument("--priors", type=none_or_str, default=None,
    #    required=False, help="optional redshift prior file")

    parser.add_argument("--chi2_scan", type=none_or_str, default=None,
        required=False, help="Load the file containing already computed chi2 scan")

    parser.add_argument("--debug", type=str2bool,  default=False,
        required=False, help="debug with ipython (only if communicator has a "
        "single process)")

    parser.add_argument("--nminima", type=int, default=3,
        required=False, help="the number of redshift minima to search")

    parser.add_argument("--cache_Rcsr", type=str2bool,  default=True,
        required=False, help="SCache Rcsr")

    parser.add_argument("--model", type=str, required=True,
        help="File pointing to the trained classifier. Required if --mode is not 'training' or 'merge'.")

    parser.add_argument("--prob_cut", default=0.0, type=float,
        help="Only objects with probability >= PROB_CUT will be included in the catalogue")

    parser.add_argument("--output_catalogue", required=True, type=str,
        help="Name of the fits file where the final catalogue will be stored. "
             "If a full path is not provided, then use --outpath")

    parser.add_argument("--quiet", type=str2bool, default=False,
        help="Do not print messages")

    parser.add_argument("--clean_dir", type=str2bool, default=True,
        help="Clean the directory of intermediate files")

    args = parser.parse_args()

    # manage verbosity
    userprint = verboseprint if not args.quiet else quietprint

    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.size
        comm_rank = comm.rank

    # Check arguments- all processes have this, so just check on the first
    # process

    if comm_rank == 0:
        if args.debug and comm_size != 1:
            print("--debug can only be used if the communicator has one "
                " process")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

    wlranges = None
    if args.wlranges[0] is not None:
        wlranges=[]
        for i in range(len(args.infiles)):
            wlranges.append([float(x) for x in args.wlranges[i].split(",")])


    arms_ratio = None
    if args.arms_ratio is not None:
        arms_ratio = [ float(x) for x in args.arms_ratio.split(",") ]
        assert len(arms_ratio) == len(args.infiles) , 'lenghtes of arms_ratio(s) and infiles must be identical'

    ### Now we have both infiles and wlranges array. We use  l1_fileinfo function to update these two parameters and join_arms
    ### and puth them in the right order, if needed. However, we had similar test done by APSOB
    infiles_info = l1_fileinfo(args.infiles, wlranges=wlranges, arms_ratio=arms_ratio)
    args.infiles = infiles_info['infiles']
    wlranges     = infiles_info['wlranges']
    arms_ratio   = infiles_info['arms_ratio']

    ## We also update the args.wlranges and args.arms_ratio for reporting purpose
    args.wlranges = wlranges
    args.arms_ratio = arms_ratio

    ## and check if join_arms is consistent with our new condition after all corrections by l1_fileinfo
    ## However, we have another check by APSOB that do the same
    if len(args.infiles) < 2:
        args.join_arms = False

    aps_ids = None
    if args.aps_ids is not None:
        aps_ids = [ int(x) for x in args.aps_ids.split(",") ]

    targsrvy = None
    if args.targsrvy is not None:
        targsrvy = [ str(x) for x in args.targsrvy.split(",") ]

    targclass = None
    if args.targclass is not None:
        targclass = [ str(x) for x in args.targclass.split(",") ]

    mask_aps_ids = None
    if args.mask_aps_ids is not None:
        mask_aps_ids = [ int(x) for x in args.mask_aps_ids.split(",")]


    area = None
    if args.area is not None:
        area = [float(x) for x in args.area.split(",")]
        ## also update the args.area to be printed in the final format in the report file
        args.area=area


    mask_areas = None
    if args.mask_areas[0] is not None:
        mask_areas=[]
        for i in range(len(args.mask_areas)):
            mask_areas.append([float(x) for x in args.mask_areas[i].split(",")])
        ## also update the args.mask_areas to be printed in the final format in the report file
        args.mask_areas=mask_areas


    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ', '')


    figdir=None
    if args.fig:
        figdir=args.outpath + '/figs/'
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            print("FIGDIR: %s Created!" %(figdir))
        figdir=figdir+'/'#+str(args.headname)
        figdir=figdir.replace(' ', '')

    # now we make some checks for SQUEzE arguments
    # first we make sure the catalogue's folder exists
    if args.output_catalogue.startswith("/"):
        catalogue_path = args.output_catalogue[:args.output_catalogue.rfind("/")]
        if not os.path.exists(catalogue_path):
            os.makedirs(catalogue_path)
            print("OUTPUT_CATALOGUE: %s Created!" %(catalogue_path))
    else:
        args.output_catalogue = os.path.join(args.outpath, args.output_catalogue)
    # and then we make sure the model used exists
    assert os.path.exists(args.model)
    model = args.model
    # finally check that prob is a number between 0 and 1
    assert (args.prob_cut >= 0.0 and args.prob_cut <= 1.0)

    # print args and assigned/default values on the screen
    print_args(args, module='SQUEzE', version=aps_constants.__aps_squeze_version__, path=outpath, headname=args.headname)

    zbest_fname=os.path.join(args.outpath,'zbest_'+str(args.headname)+'.fits')

    zall_fname=None
    if args.zall:
        zall_fname=os.path.join(args.outpath,'zall_'+str(args.headname)+'.fits')
        zall_fname=zall_fname.replace(' ', '')


    zspec_fname=os.path.join(args.outpath,'zspec_'+str(args.headname)+'.fits')
    zspec_fname=zspec_fname.replace(' ', '')

    if (not args.overwrite) and os.path.exists(zbest_fname):
        print('skipping, products already exist. If this is a test run, change the --headname')
        sys.exit()


    ### Before running the main worker, we first make sure eveyrthing is OK
    try:
        infiles_check = l1_fileinfo(args.infiles)
        fcheck_targs, fcheck_idt, fcheck_info, fcheck_la = gen_targlist(infiles_check['infiles'][0],
            infiles_check['mode'], aps_ids = aps_ids, targsrvy= targsrvy, targclass = targclass,
            mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, la_out=False)

        # ## Make sure at least one valid aps_id exist after applying all filters
        assert len(fcheck_targs) > 0, 'No valid APS_ID(s) found'


        status_fibs = [fcheck_info['FIB_STATUS'][fcheck_idt[fchid]] for fchid in fcheck_targs]

        ## Count number of targets with ACTIVE fibres
        ## Note, we also check if fib_status of fibres are OK through 1- the main rrweave_worker where we read data and
        ## 2- where we pack data into the redrock in the pack_2_redrock function
        n_active_fibs = status_fibs.count('A')

        ## Make sure at least one valid aps_id exist (with fib_status = A: ACTIVE) after applying all filters
        assert n_active_fibs > 0, 'FIB_STATUS CHECK: No valid APS_ID(s) found'


        ## Update args.mp if it was greater than number of targets (with valid fib-type)
        if n_active_fibs < args.mp:
            args.mp = n_active_fibs
            print('NOTE: --mp param is updated to %d'%(args.mp))
    except:
        print('ERROR : %s' %(sys.exc_info()[1]))
        return

    # run SQUEzE
    assert (args.output_catalogue.endswith(".fit") or
            args.output_catalogue.endswith(".fits") or
            args.output_catalogue.endswith(".fits.gz")
            ), "Invalid extension for output catalogue"
    ext = args.output_catalogue[args.output_catalogue.rfind("fit")-1:]
    save_file = args.output_catalogue.replace(ext,
                                              "_squeze_candidates{}".format(ext))
    if not save_file.startswith("/"):
        save_file = args.outpath + save_file

    candidates_df, z_precision = squeze_worker(args.infiles, args.model,
                                               aps_ids, targsrvy, targclass,
                                               mask_aps_ids, area, mask_areas,
                                               wlranges, args.cache_Rcsr,
                                               args.sens_corr, args.mask_gaps,
                                               args.vacuum, args.tellurics,
                                               args.fill_gap, arms_ratio,
                                               args.join_arms,
                                               quiet=args.quiet,
                                               save_file=save_file)

    # here is where we format SQUEzE output into priors
    # we currently take a flat prior with SQUEzE preferred redshift solution
    # and a width as specified in the redshift precision used to train
    # SQUEzE model
    priors = args.outpath + "/priors.fits.gz"
    aux = candidates_df[(~candidates_df["DUPLICATED"]) &
                                  (candidates_df["PROB"] >= args.prob_cut)]
    columns = [
        fits.Column(name="TARGETID",
                    format="I",
                    array=aux["APS_ID"]),
        fits.Column(name="Z",
                    format="D",
                    array=aux["Z_TRY"]),
        fits.Column(name="SIGMA",
                    format="D",
                    array=np.ones(aux.shape[0])*z_precision),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="PRIORS")
    hdu.writeto(priors, overwrite=args.overwrite)
    # update aps_ids to only include objects with prior
    if aps_ids is None:
        aps_ids = list(aux["APS_ID"])
    else:
        for aps_id in aps_ids:
            if not apd_id in aux["APS_ID"]:
                aps_ids.pop(aps_id)
    del aux, columns, hdu

    # now we run redrock
    scandata, zbest, zspec, zfitall = rrweave_worker(
        args.infiles, args.templates, args.srvyconf,
        zbest_fname=zbest_fname, zall_fname =zall_fname,zspec_fname=zspec_fname, aps_ids=aps_ids,
        targsrvy= targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=args.sens_corr, mask_gaps=args.mask_gaps, vacuum=args.vacuum, tellurics=args.tellurics,
        fill_gap=args.fill_gap, arms_ratio=arms_ratio, join_arms=args.join_arms, ncpus= args.mp, comm=comm,
        comm_rank=comm_rank, comm_size=comm_size,nminima=args.nminima, archetypes=args.archetypes,cache_Rcsr=args.cache_Rcsr,
        priors=priors, chi2_scan=args.chi2_scan, figdir=figdir, debug=args.debug, return_outputs=True)

    # here we format results according to the CS specifications
    # and save the catalogues
    write_results(zbest, candidates_df, args)

    # clean the directory
    if args.clean_dir:
        if os.path.exists(priors):
            os.remove(priors)
        if os.path.exists(save_file):
            os.remove(save_file)
        if os.path.exists("{}/zall_test.fits".format(args.output_path)):
            os.remove("{}/zall_test.fits".format(args.output_path))
        if os.path.exists("{}/zbest_test.fits".format(args.output_path)):
            os.remove("{}/zbest_test.fits".format(args.output_path))
        if os.path.exists("{}/zspec_test.fits".format(args.output_path)):
            os.remove("{}/zspec_test.fits".format(args.output_path))
    # TODO: clean redrock results if necessary

    userprint("Done")

##########################################################
if __name__ == '__main__':

    option = [
    '--infiles', '/Users/iperezra/software/SQUEzE/py/squeze/tests/data/stacked_1004073.fit', '/Users/iperezra/software/SQUEzE/py/squeze/tests/data/stacked_1004073.fit',
    '--aps_ids', '1004,1003, 1002,9,1007',
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area', 'None',
    '--mask_areas', 'None',
    '--wlranges', '4800.0,5000', '6000.0,6100',
    '--sens_corr', 'True',
    '--mask_gaps', 'True',
    '--tellurics', 'False',
    '--vacuum', 'True',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0,0.83',
    '--join_arms', 'True',
    '--templates', '/Users/iperezra/software/SQUEzE/py/squeze/tests/data/templates_RR/',
    '--srvyconf', '/Users/alireza/PyAPS/config_files/weave_cls.json',
    '--archetypes', '/Users/iperezra/software/SQUEzE/py/squeze/tests/data/redrock-archetypes/',
    '--outpath', '/Users/alireza/PyAPS_results/20160907/3161/',
    '--headname', 'test',
    '--zall', 'True',
    '--chi2_scan', 'None',
    '--nminima' , '3',
    '--fig', 'True',
    '--cache_Rcsr', 'False',
    '--debug', 'False',
    '--overwrite', 'True',
    '--mp' ,'2',
    '--model', '/Users/iperezra/software/SQUEzE/data/BOSS_train_64plates_model.json',
    "--prob_cut", '0.0',
    "--output_catalogue", '',
    ]

    # Set option to None to run the code through the command line
    option = None

    main(options=option, comm=None)
##########################################################
