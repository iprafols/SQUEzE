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
                          |                 |  --quiet            (Optional)      |        False      |                               |
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

##########################################################

def write_ztable(outfile, ztable, template_version, archetype_version):
    """Write output Table to outfile

    Args:
        outfile (str): output file.
        ztable (Table): input table (astropy).
    """
    header = fits.Header()
    header['RR_V'] = (__version__, 'Redrock version')
    header['APS_V'] = (aps_constants.__aps_version__,'PyAPS version')
    header['APS_RR_V'] = (aps_constants.__aps_rr_version__,'PyAPS Redrock wrapper version')

    ztable.meta['RR_V'] = (__version__, 'Redrock version')
    ztable.meta['APS_V'] = (aps_constants.__aps_version__,'PyAPS version')
    ztable.meta['APS_RR_V'] = (aps_constants.__aps_rr_version__,'PyAPS Redrock wrapper version')

    # header['MIN_WAVE'] = lmin
    # header.comments['MIN_WAVE']= 'Min wavelength (Ang), used by APS_REDROCK'
    # header['MAX_WAVE'] = lmax
    # header.comments['MAX_WAVE']= 'MAX wavelength (Ang), used by APS_REDROCK'

    for i, fulltype in enumerate(template_version.keys()):
        header['TEMNAM'+str(i).zfill(2)] = fulltype
        header['TEMVER'+str(i).zfill(2)] = template_version[fulltype]

        ztable.meta['TEMNAM'+str(i).zfill(2)] = fulltype
        ztable.meta['TEMVER'+str(i).zfill(2)] = template_version[fulltype]

    if not archetype_version is None:
        for i, fulltype in enumerate(archetype_version.keys()):
            header['ARCNAM'+str(i).zfill(2)] = fulltype
            header['ARCVER'+str(i).zfill(2)] = archetype_version[fulltype]

            ztable.meta['ARCNAM'+str(i).zfill(2)] = fulltype
            ztable.meta['ARCVER'+str(i).zfill(2)] = archetype_version[fulltype]

    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU(header=header))
    hx.append(fits.convenience.table_to_hdu(ztable))
    hx.writeto(os.path.expandvars(outfile), overwrite=True)
    return

##########################################################


def read_spectra(infiles, pack_2_redrock=False, aps_ids=None, targsrvy= None, targclass = None, mask_aps_ids = None , area=None, mask_areas=None, wlranges=None,
  cache_Rcsr=None, sens_corr=None, mask_gaps=None, vacuum=None,
   tellurics=None, fill_gap=None, arms_ratio=None, join_arms=None):

    """Read targets from a list of spectra files
    Args:
         infiles (list): input files
    Returns:
        tuple: (APStargets,APSmeta , setups) where targets is a list of Target objects and
    This setups is the output setups, after checking for availibility of data in the requested arms.
    """


    # READ DATA and put them in the APSOBJ OBJECT
    _APSOB = APSOB(infiles, aps_ids=aps_ids, targsrvy= targsrvy, targclass = targclass, mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps, vacuum=vacuum, tellurics=tellurics,
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms)

    if pack_2_redrock:
        return _APSOB.pack_2_redrock(cache_Rcsr=cache_Rcsr)
    else:
        return _APSOB

##########################################################

def gen_zfitall(zfit, apsmeta, dtemplates, darchetypes=None, zall_fname=None,comm = None):

    for colname in zfit.colnames:
        if colname.islower():
            zfit.rename_column(colname, colname.upper())

    zfit.rename_column('SPECTYPE', 'CLASS')
    zfit.rename_column('SUBTYPE', 'SUBCLASS')
    zfit.sort(['APS_ID', 'ZNUM'])

    ## Add this level, we first add TARGID, APS_ID and CNAME to the new table
    ## and then, add other columns
    ## This is just to avoid reshuffling columns, we observed in the new version of astropy and numpy
    zfitall=Table()
    zfitall.add_columns([zfit['APS_ID'],zfit['TARGID'], zfit['CNAME']])
    zfitall.add_columns([zfit['Z'], zfit['ZERR'], zfit['ZWARN'], zfit['ZNUM'], zfit['CLASS'],
    zfit['SUBCLASS'], zfit['TARGSRVY'],zfit['FIB_STATUS'], zfit['SNR'], zfit['CHI2'],
    zfit['DELTACHI2'], zfit['NCOEFF'],zfit['COEFF'],zfit['NPIXELS'],zfit['ZZ'],zfit['ZZCHI2']])


    ## Create two columns, reserved for P(z) and its error. Later it fill be filled with right values
    ## However, how it works, needs to be clarified with Dan Smath from WEAVE_LOFAR
    ## Please note, we just copied a column structure of a column in zfitall and then assigned a new new to that.
    ## We just used zfitall['z'] as reference, as it has dtype =float
    # zfitall.add_column(zfitall['Z'], name='PZ')
    # zfitall['PZ'][:] = np.NaN
    # zfitall.add_column(zfitall['Z'], name='PZERR')
    # zfitall['PZERR'][:] = np.NaN



    ## ADD META to ZBEST table
    zfitall.meta['EXTNAME'] = 'ZFITALL'
    ## ADD Original filenames (infiles) as meta to the fits file (As provinces)
    for n_province, province in enumerate(apsmeta['files']):
        zfitall.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    zfitall.meta['CSB_RR'] = (apsmeta['stitched'], 'Combines Spectral Bands Status for REDROCK')


    template_version = {t._template.full_type:t._template._version for t in dtemplates}

    archetype_version = None
    if not darchetypes is None:
        archetype_version = {name:arch._version for name, arch in darchetypes.items()}

    if zall_fname is not None:
        write_ztable(zall_fname, zfitall, template_version, archetype_version)

    return zfitall

#########################################################################################

def gen_zbest(zfitall, scandata, apsmeta, srvyconf, dtemplates, darchetypes=None, zbest_fname=None, zall_fname=None, comm = None):

    start = elapsed(None, "", comm=comm)
    ## GENERATE ZBEST TABLE
    zbest = zfitall[zfitall['ZNUM'] == 0]
    zbest.remove_columns(['ZZ', 'ZZCHI2', 'ZNUM'])




    ################################ EXPERIMENTAL ########################################
    ## add new class info based on the TARGSRVY keyword provided by surveys.
    ## It Read a file, called WEAVE_CLS.json, contain a dic of class for different surveys
    ##
    ## TODO list (1/8/2019): List should be updated/modified. Code must be rewrite in a more beutifull shape.

    zbest.add_column(zbest['CLASS'], name='SRVY_CLASS')

    if not os.path.exists(srvyconf):
        sys.exit()

    WSRVYS = json.load(open(srvyconf))
    # Note: it just compare between first two char of TARGSRV and WSRVYS dict
    for wsc in range(0, len(zbest)):
        targsrv= str(zbest['TARGSRVY'][wsc])[:2].replace(' ','').upper()
        if targsrv in [x[:2].upper() for x in WSRVYS['GALAXY']] : zbest['SRVY_CLASS'][wsc] = np.str('GALAXY')
        if targsrv in [x[:2].upper() for x in WSRVYS['STAR']]   : zbest['SRVY_CLASS'][wsc] = np.str('STAR')
        if targsrv in [x[:2] for x in WSRVYS['QSO']]   : zbest['SRVY_CLASS'][wsc] = np.str('QSO')
        # if targsrv in [x[:2] for x in WSRVYS['NEBULA']]   : zbest['SRVY_CLASS'][wsc] = 'NEBULA'
        # if targsrv in [x[:2] for x in WSRVYS['PI']]   : zbest['SRVY_CLASS'][wsc] = 'PI'
    #########################################################################################

    ## Filter object with FIB_STATUS !=A (Allocated fibre = not parked or broken)
    ## However, by default we check this while packing data into the redrock format (pack_2_redrock function)
    ## and also before running the main rrweave_worker in this routine, where we check the input parameters
    ## FOR targets with STATUS !=A, We change CLASS and SUBCLASS of a copy of ZBEST to 'NAN'
    ## We create a copy of ZBEST table (called zbest_original), as any change to the main zbest table
    ## at this level, will affect the zspec table, where we go through it in the next steps.

    # zbest_original = Table(zbest)

    # for tabid in range(0, len(zbest)):
    #     if zbest['FIB_STATUS'][tabid].upper() != 'A':
    #         zbest['CLASS'][tabid]='nan'
    #         zbest['SUBCLASS'][tabid]='nan'
    #         zbest['SRVY_CLASS'][tabid]='nan'

    ## after the previous step, we are done with FIB_STATUS column, so we remove it from final zbest table
    zbest.remove_columns(['FIB_STATUS'])

     ## ADD META to ZBEST table
    zbest.meta['EXTNAME'] = 'ZBEST'
    ## ADD Original filenames (infiles) as meta to the fits file (As provinces)
    for n_province, province in enumerate(apsmeta['files']):
        zbest.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    zbest.meta['CSB_RR'] = (apsmeta['stitched'], 'Combines Spectral Bands Status for REDROCK')



    ## Add results for chi2(Z) on coarse grids for all availbale templates to the zbest table
    try:
        scan_table = gen_scandata_table(scandata)
        zbest = join(zbest, scan_table, join_type='left', keys='APS_ID')
    except:
        print('Failed to proceed with the CORSE REDSHIFT SEARCH CHI2(Z) implementation')

    print(zbest)

    if zbest_fname is not None:

        template_version = {t._template.full_type:t._template._version for t in dtemplates}
        archetype_version = None
        if not darchetypes is None:
            archetype_version = {name:arch._version for name, arch in darchetypes.items() }

        write_ztable(zbest_fname, zbest, template_version, archetype_version)

    stop = elapsed(start, "Preparing/writing ZBEST (and ZALL) tables took", comm=comm)
    # return zbest, zbest_original
    return zbest

#########################################################################################

def gen_zspec(zbest, targets, apsmeta, dtemplates, darchetypes=None, sens_corr = True, zspec_fname=None, comm = None):
    ## START with making zspec table, including the raw spectra and best fits
    start_zspec = elapsed(None, "", comm=comm)

    template_version = {t._template.full_type:t._template._version for t in dtemplates}

    archetype_version = None
    if not darchetypes is None:
        archetype_version = {name:arch._version for name, arch in darchetypes.items() }


    # add template full_type to the dtemplates, already generated
    # just to call them, based on the full_type (GALAXY, QSO, STAR::A, etc)
    templates = dict()
    for dt in dtemplates:
        templates[dt.template.full_type] = dt.template

    ### GENERATE OUTPUT TABLE STRUTURE
    columns = ['APS_ID','TARGID','CNAME']#,'Z', 'ZERR', 'ZWARN', 'CLASS', 'SUBCLASS', 'SNR']

    # for subs in range(0, len(dwave)):

    # loop over all availble setups (those generated by the code, not neccesariliy those in input param)
    for s in apsmeta['setups']:
        columns.append('LAMBDA_RR_%s' % s[0])
        columns.append('FLUX_RR_%s' % s[0])
        columns.append('IVAR_RR_%s' % s[0])
        columns.append('MODEL_RR_%s' % s[0])

    outdict = OrderedDict()
    for c in columns:
        outdict[c]=[]

    #####################################

    for i in range(0, len(targets)):
        ## first we make sure all targets, coming from dtargets.local() are unpacked from shared_memory
        targets[i].sharedmem_unpack()
        dtg = targets[i]
        id_zbest=np.where(zbest['APS_ID'] == dtg.id)[0]

        if not id_zbest.size == 0:
            zz=zbest[id_zbest]
            coeff = zz['COEFF'].reshape(-1)
            fulltype = zz['CLASS'][0]

            if zz['SUBCLASS'] != '':
                fulltype = fulltype+':::'+zz['SUBCLASS'][0]

            if darchetypes is not None:
                dwave = { s.wavehash:s.wave for s in dtg.spectra}
                tp = darchetypes[zz['CLASS'][0]]
            else:
                tp = templates[fulltype]
                if tp.template_type != zz['CLASS']:
                    raise ValueError('CLASS TYPE: {} not in'
                        ' templates'.format(zz['CLASS']))


            outdict['APS_ID'].append(zbest[id_zbest]['APS_ID'][0])
            outdict['TARGID'].append(zbest[id_zbest]['TARGID'][0])
            outdict['CNAME'].append(zbest[id_zbest]['CNAME'][0])
            # outdict['Z'].append(zbest[id_zbest]['Z'][0])
            # outdict['ZERR'].append(zbest[id_zbest]['ZERR'][0])
            # outdict['ZWARN'].append(zbest[id_zbest]['ZWARN'][0])
            # outdict['CLASS'].append(np.str(zbest[id_zbest]['CLASS'][0]))
            # outdict['SUBCLASS'].append(np.str(zbest[id_zbest]['SUBCLASS'][0]))
            # outdict['SNR'].append(zbest[id_zbest]['SNR'][0])


            for dwc, s in enumerate(apsmeta['setups']):

                if darchetypes is not None:
                    model = tp.eval(zz['SUBCLASS'], dwave, coeff, dtg.spectra[dwc].wave, zz['Z']) * (1+zz['Z'])
                else:
                    model=tp.eval(coeff[0:tp.nbasis], dtg.spectra[dwc].wave, zz['Z']) * (1+zz['Z'])

                model = dtg.spectra[dwc].R.dot(model)

                flux = dtg.spectra[dwc].flux.copy()
                wave = dtg.spectra[dwc].wave.copy()
                ivar = dtg.spectra[dwc].ivar.copy()
                isbad = (ivar == 0)
                flux[isbad] = np.NaN
                model[isbad] = np.NaN
                outdict['LAMBDA_RR_%s' % s[0]].append(wave)
                outdict['FLUX_RR_%s' % s[0]].append(flux)
                outdict['IVAR_RR_%s' % s[0]].append(ivar)
                outdict['MODEL_RR_%s' % s[0]].append(model)

    zspec = Table(outdict)
    zspec.sort(['APS_ID'])
    zspec.meta['EXTNAME']='CLASS_SPECTRA'
    ## ADD Original filenames (infiles) as meta to the fits file (As provinces)
    for n_province, province in enumerate(apsmeta['files']):
        zspec.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')


    ## add Units to flux, model and ivar in the zspec table
    if sens_corr:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' %(apsmeta['funits'])
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' %(apsmeta['funits']**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'

    for s in apsmeta['setups']:
        zspec['LAMBDA_RR_%s' % s[0]].unit= wave_unit_str
        zspec['FLUX_RR_%s' % s[0]].unit= flux_unit_str
        zspec['IVAR_RR_%s' % s[0]].unit= ivar_unit_str
        zspec['MODEL_RR_%s' % s[0]].unit= flux_unit_str

    ## add a keyword to the header, indicating the Wavelengths are in vacuum, sampling and the latest status of stitching
    zspec.meta['VACUUM'] = (True, 'Wavelengths are in vacuum')
    zspec.meta['SAMPLING'] = (0,'Sampling mode (0: linear, 1: logarithmic')
    zspec.meta['CSB_RR'] = (apsmeta['stitched'], 'Combines Spectral Bands Status for REDROCK')


    ## write zspec tabloe into a fits file
    if zspec_fname is not None:
        write_ztable(zspec_fname, zspec, template_version, archetype_version)
    stop = elapsed(start_zspec, "Writing ZSPEC fits files took", comm=comm)

    return zspec

##########################################################

def rrweave_worker(infiles, templates, srvyconf, zbest_fname= None, zall_fname = None ,zspec_fname=None, aps_ids=None,
    targsrvy= None, targclass = None, mask_aps_ids=None , area=None, mask_areas=None, wlranges=None, sens_corr=True,
    mask_gaps=True, vacuum=True, tellurics=False, fill_gap=False, arms_ratio=None, join_arms=False, ncpus= 1, comm=None,
    comm_rank=0, comm_size=1, nminima=3, archetypes=None, cache_Rcsr= False, priors=None, chi2_scan=None, figdir=None,
    debug=False, return_outputs=False, binmode = 'FIB'):

    """
    Function description:

    """


    global_start = elapsed(None, "", comm=comm)
    # Multiprocessing processes to use if MPI is disabled.
    mpprocs = 0
    if comm is None:
        mpprocs = get_mp(ncpus)
        print("Running with {} processes".format(mpprocs))
        if "OMP_NUM_THREADS" in os.environ:
            nthread = int(os.environ["OMP_NUM_THREADS"])
            if nthread != 1:
                print("WARNING:  {} multiprocesses running, each with "
                    "{} threads ({} total)".format(mpprocs, nthread,
                    mpprocs*nthread))
                print("WARNING:  Please ensure this is <= the number of "
                    "physical cores on the system")
        else:
            print("WARNING:  using multiprocessing, but the OMP_NUM_THREADS")
            print("WARNING:  environment variable is not set- your system may")
            print("WARNING:  be oversubscribed.")
        sys.stdout.flush()
    elif comm_rank == 0:
        print("Running with {} processes".format(comm_size))
        sys.stdout.flush()

    try:
        # Load and distribute the targets
        if comm_rank == 0:
            print("Loading targets (according to thier APS_IDs)...")
            sys.stdout.flush()

        start = elapsed(None, "", comm=comm)
        # Read the spectra on the root process.  Currently the "meta" Table
        # returned here is not propagated to the output zbest file.  However,
        # that could be changed to work like the DESI write_zbest() function.
        # Each target contains metadata which is propagated to the output zbest
        # table though.

        ## Added 17 Feb 2020: we added a parameter called binmode to make it possible to run the rrweave_worker on
        ## voronoi-bins. To use this set the binmode to 'BIN'
        if binmode.upper().replace(" ","") == 'FIB':
            targets, apsmeta = read_spectra(infiles, aps_ids= aps_ids, targsrvy= targsrvy, targclass = targclass,
                mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, wlranges=wlranges, cache_Rcsr=cache_Rcsr,
                sens_corr=sens_corr, mask_gaps=mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap,
                arms_ratio=arms_ratio, join_arms=join_arms, pack_2_redrock=True)

        if binmode.upper().replace(" ","") == 'BIN':
            print('LATER WE WILL ADD A FUCTION TO READ BINS (VORONOI BINS)')
            sys.exit()

        stop = elapsed(start, "Read of {} targets".format(len(targets)), comm=comm)


        # Distribute the targets.
        start = elapsed(None, "", comm=comm)
        dtargets = DistTargetsCopy(targets, comm=comm, root=0)

        # Get the dictionary of wavelength grids
        dwave = dtargets.wavegrids()
        stop = elapsed(start, "Distribution of {} targets"\
            .format(len(dtargets.all_target_ids)), comm=comm)

        # Read the template data
        dtemplates = load_dist_templates(dwave, templates=templates,
            comm=comm, mp_procs=mpprocs)


        # Read archetypes data if availble
        if not archetypes is None:
            start_arctype = elapsed(None, "", comm=comm)
            darchetypes = All_archetypes(archetypes_dir=archetypes).archetypes
            stop = elapsed(start_arctype, "Preparing archetypes data took", comm=comm)
        else:
            darchetypes = None


        # Compute the redshifts, including both the coarse scan and the
        # refinement.  This function only returns data on the rank 0 process.
        start_redshift = elapsed(None, "", comm=comm)
        scandata, zfit = zfind(dtargets, dtemplates, mp_procs=mpprocs, nminima=nminima, archetypes = archetypes, priors=priors, chi2_scan=chi2_scan)
        stop = elapsed(start_redshift, "Computing redshifts took", comm=comm)

        # print(scandata)

        ### PREPARING/GENERATING outputs
        start_tables = elapsed(None, "", comm=comm)
        if comm_rank == 0:

            zfitall = gen_zfitall(zfit, apsmeta, dtemplates, darchetypes = darchetypes, zall_fname=zall_fname)

            # zbest, zbest_original = gen_zbest(zfitall, apsmeta, srvyconf, dtemplates,
            #     darchetypes=darchetypes,zbest_fname=zbest_fname, zall_fname=zall_fname)

            zbest = gen_zbest(zfitall, scandata, apsmeta, srvyconf, dtemplates,
                darchetypes=darchetypes,zbest_fname=zbest_fname, zall_fname=zall_fname)


            ## As zbest has been modified (change class of targets with fib_stat !='A' to nan), for the following step
            ## we use the original zbest table (zbest_original)

            # zspec = gen_zspec(zbest_original, dtargets.local(), apsmeta, dtemplates, darchetypes= darchetypes, sens_corr = sens_corr, zspec_fname=zspec_fname)
            zspec = gen_zspec(zbest, dtargets.local(), apsmeta, dtemplates, darchetypes= darchetypes, sens_corr = sens_corr, zspec_fname=zspec_fname)

        stop = elapsed(start_tables, "Preparing ALL output tables/fits files took", comm=comm)


        ## START generating the figures, demonestrating the raw spectra and the best fits
        if (figdir is not None) and (comm_rank == 0):
            start_fig = elapsed(None, "", comm=comm)
            try:
                make_rrplot(zbest, zspec, apsmeta['setups'],  figdir)
            except:
                print('Failed to generate plots. 1- Check your X11 configs. 2- Check spectra in output fits file.')
                pass

            stop = elapsed(start_fig, "Generating figures took", comm=comm)

    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "Proc {}: {}".format(comm_rank, x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()
        if comm is not None:
            comm.Abort()

    global_stop = elapsed(global_start, "Total run time", comm=comm)

    if debug:
        import IPython
        IPython.embed()

    if return_outputs:
        return scandata, zbest, zspec, zfitall
    else:
        return


##########################################################

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
    weave_formatted_spectra = read_spectra(infiles,
                                           aps_ids= aps_ids,
                                           targsrvy= targsrvy,
                                           targclass = targclass,
                                           mask_aps_ids = mask_aps_ids,
                                           area=area,
                                           mask_areas=mask_areas,
                                           wlranges=wlranges,
                                           cache_Rcsr=cache_Rcsr,
                                           sens_corr=sens_corr,
                                           mask_gaps=mask_gaps,
                                           vacuum=vacuum,
                                           tellurics=tellurics,
                                           fill_gap=fill_gap,
                                           arms_ratio=arms_ratio,
                                           join_arms=join_arms,
                                           pack_2_redrock=False)

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

def write_results(zfitall, candidates, args):
    """ Format results according to the specifications of the CS
    specifications

    Args:
        zfitall (astropy.TABLE): Table with the redrock results run with SQUEzE priors
        candidates (pd.DataFrame): DataFrame with SQUEzE candidates' information
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

    # then create the final catalogue
    nonduplicate_candidates = candidates[(~candidates["DUPLICATED"])]
    cnames = nonduplicate_candidates["CNAME"]
    targids = nonduplicate_candidates["TARGID"]
    targclasses = ["QSO"]*cnames.size
    probs = nonduplicate_candidates["PROB"]
    for cname in cnames:
        z = zfitall[zfitall["CNAME"] == cname]["Z"]
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
                    format="7A",
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
    cnames = nonduplicate_candidates["CNAME"]
    probs = np.zeros(num_peaks)
    z = np.zeros(num_peaks)
    for cname in cnames:
        candidates_per_cname = candidates[candidates["CNAME"] == cname]
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

    parser.add_argument("--quiet", action="store_true",
        help="Do not print messages")

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
    if args.debug:
        assert (args.output_catalogue.endswith(".fit") or
                args.output_catalogue.endswith(".fits") or
                args.output_catalogue.endswith(".fits.gz")
                ), "Invalid extension for output catalogue"
        ext = args.output_catalogue[args.output_catalogue.rfind("fit")-1:]
        save_file = args.output_catalogue.replace(ext,
                                                  "_squeze_candidates{}".format(ext))
        if not save_file.startswith("/"):
            save_file = args.outpath + save_file
    else:
        save_file = None

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
                    format="15A",
                    array=aux["TARGID"]),
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
    write_results(zfitall, candidates_df, args)

    # clean the directory
    if not args.debug and os.path.exists(priors):
        os.remove(priors)
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
