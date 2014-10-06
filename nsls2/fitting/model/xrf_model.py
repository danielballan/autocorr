# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 09/10/2014                                                #
#                                                                      #
# Original code:                                                       #
# @author: Mirna Lerotic, 2nd Look Consulting                          #
#         http://www.2ndlookconsulting.com/                            #
# Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory         #
# All rights reserved.                                                 #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import six

import logging
logger = logging.getLogger(__name__)

from nsls2.constants import Element
from nsls2.fitting.model.physics_peak import (gauss_peak)
from nsls2.fitting.model.physics_model import (ComptonModel, ElasticModel,
                                               _gen_class_docs)
from nsls2.fitting.base.parameter_data import get_para
from lmfit import Model


k_line = ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
          'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
          'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
          'In', 'Sn', 'Sb', 'Te', 'I', 'dummy', 'dummy']

l_line = ['Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L', 'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L', 'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
          'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L', 'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L', 'Au_L', 'Hg_L', 'Tl_L',
          'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L', 'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L', 'Br_L', 'Ga_L']

m_line = ['Au_M', 'Pb_M', 'U_M', 'noise', 'Pt_M', 'Ti_M', 'Gd_M', 'dummy', 'dummy']


def gauss_peak_xrf(x, area, center,
                   delta_center, delta_sigma,
                   ratio, ratio_adjust,
                   fwhm_offset, fwhm_fanoprime,
                   e_offset, e_linear, e_quadratic,
                   epsilon=2.96):
    """
    This is a function to construct xrf element peak, which is based on gauss profile,
    but more specific requirements need to be considered. For instance, the standard
    deviation is replaced by global fitting parameters, and energy calibration on x is
    taken into account.

    Parameters
    ----------
    x : array
        independent variable
    area : float
        area of gaussian function
    center : float
        center position
    delta_center : float
        adjustment to center position
    delta_sigma : float
        adjustment to standard deviation
    ratio : float
        branching ratio
    ratio_adjust : float
        value used to adjust peak height
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    e_offset : float
        offset of energy calibration
    e_linear : float
        linear coefficient in energy calibration
    e_quadratic : float
        quadratic coefficient in energy calibration

    Returns
    -------
    array:
        gaussian peak profile
    """
    def get_sigma(center):
        temp_val = 2 * np.sqrt(2 * np.log(2))
        return np.sqrt((fwhm_offset/temp_val)**2 + center*epsilon*fwhm_fanoprime)

    x = e_offset + x * e_linear + x**2 * e_quadratic

    return gauss_peak(x, area, center+delta_center,
                      delta_sigma+get_sigma(center)) * ratio * (1 + ratio_adjust)


class GaussModel_xrf(Model):

    __doc__ = _gen_class_docs(gauss_peak_xrf)

    def __init__(self, *args, **kwargs):
        super(GaussModel_xrf, self).__init__(gauss_peak_xrf, *args, **kwargs)
        self.set_param_hint('epsilon', value=2.96, vary=False)


def _set_parameter_hint(para_name, input_dict, input_model,
                        log_option=False):
    """
    Set parameter information to a given model

    Parameters
    ----------
    para_name : str
        parameter used for fitting
    input_dict : dict
        all the initial values and constraints for given parameters
    input_model : object
        model object used in lmfit
    log_option : bool
        option for logger
    """

    if input_dict['bound_type'] == 'none':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True)
    elif input_dict['bound_type'] == 'fixed':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=False)
    elif input_dict['bound_type'] == 'lohi':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                   min=input_dict['min'], max=input_dict['max'])
    elif input_dict['bound_type'] == 'lo':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                   min=input_dict['min'])
    elif input_dict['bound_type'] == 'hi':
        input_model.set_param_hint(name=para_name, value=input_dict['value'], vary=True,
                                   min=input_dict['max'])
    else:
        raise ValueError("could not set values for {0}".format(para_name))
    if log_option:
        logger.info(' {0} bound type: {1}, value: {2}, range: {3}'.
                    format(para_name, input_dict['bound_type'], input_dict['value'],
                           [input_dict['min'], input_dict['max']]))
    return


def update_parameter_dict(xrf_parameter, fit_results, bound_option):
    """
    Update fitting parameters according to previous fitting results.

    Parameters
    ----------
    xrf_parameter : dict
        saving all the fitting values and their bounds
    fit_results : object
        ModelFit object from lmfit
    bound_option : str
        define bound type

    Returns
    -------
    dict
        updated xrf parameters
    """

    new_parameter = xrf_parameter.copy()
    for k, v in six.iteritems(new_parameter):
        if k == 'element_list':
            continue
        if k.startswith('ratio'):
            itemv = k.split('-')
            k_new = itemv[1]+'_'+itemv[2]+'_ratio_adjust'
        elif k.startswith('pos'):
            itemv = k.split('-')
            k_new = itemv[1]+'_'+itemv[2]+'_delta_center'
        elif k.startswith('width'):
            itemv = k.split('-')
            k_new = itemv[1]+'_'+itemv[2]+'_delta_sigma'
        else:
            k_new = k

        v['bound_type'] = v[str(bound_option)]

        if k_new in list(fit_results.values.keys()):
            v['value'] = fit_results.values[str(k_new)]
    return new_parameter


def add_element_dict(xrf_parameter, element_list=None):

    new_parameter = xrf_parameter.copy()

    if element_list is None:
        if ',' in xrf_parameter['element_list']:
            element_list = xrf_parameter['element_list'].split(', ')
        else:
            element_list = xrf_parameter['element_list'].split()
        element_list = [item.strip() for item in element_list]

    for item in element_list:
        if item in k_line:
            pos_add_ka1 = {"pos-"+str(item)+"-ka1":
                               {"bound_type": "fixed", "min": -0.005, "max": 0.005, "value": 0,
                                "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed"}}

            width_add_ka1 = {"width-"+str(item)+"-ka1":
                                 {"bound_type": "fixed", "min": -0.02, "max": 0.02, "value": 0.0,
                                  "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed"}}

            pos_add_ka2 = {"pos-"+str(item)+"-ka2":
                               {"bound_type": "fixed", "min": -0.01, "max": 0.01, "value": 0,
                                "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed"}}

            width_add_ka2 = {"width-"+str(item)+"-ka2":
                                 {"bound_type": "fixed", "min": -0.02, "max": 0.02, "value": 0.0,
                                  "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed"}}

            pos_add_kb1 = {"pos-"+str(item)+"-kb1":
                               {"bound_type": "fixed", "min": -0.01, "max": 0.01, "value": 0,
                                "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed"}}

            width_add_kb1 = {"width-"+str(item)+"-kb1":
                                 {"bound_type": "fixed", "min": -0.02, "max": 0.02, "value": 0.0,
                                  "free_more": "fixed", "adjust_element": "lohi", "e_calibration": "fixed"}}

            add_list = [pos_add_ka1, width_add_ka1,
                        pos_add_ka2, width_add_ka2,
                        pos_add_kb1, width_add_kb1]

            for addv in add_list:
                new_parameter.update(addv)
    return new_parameter


class ModelSpectrum(object):

    def __init__(self, xrf_parameter):
        """
        Parameters
        ----------
        xrf_parameter : dict
            saving all the fitting values and their bounds
        """

        self.parameter = xrf_parameter

        if self.parameter.has_key('element_list'):
            if ',' in self.parameter['element_list']:
                self.element_list = self.parameter['element_list'].split(', ')
            else:
                self.element_list = self.parameter['element_list'].split()
            self.element_list = [item.strip() for item in self.element_list]
        else:
            logger.critical(' No element is selected for fitting!')

        self.incident_energy = self.parameter['coherent_sct_energy']['value']

        self.parameter_default = get_para()
        return

    def setComptonModel(self):
        """
        setup parameters related to Compton model
        """
        compton = ComptonModel()

        compton_list = ['coherent_sct_energy', 'compton_amplitude',
                        'compton_angle', 'fwhm_offset', 'fwhm_fanoprime',
                        'e_offset', 'e_linear', 'e_quadratic',
                        'compton_gamma', 'compton_f_tail',
                        'compton_f_step', 'compton_fwhm_corr',
                        'compton_hi_gamma', 'compton_hi_f_tail']

        logger.debug(' ###### Started setting up parameters for compton model. ######')
        for name in compton_list:
            if name in self.parameter.keys():
                _set_parameter_hint(name, self.parameter[name], compton)
            else:
                _set_parameter_hint(name, self.parameter_default[name], compton)
        logger.debug(' Finished setting up paramters for compton model.')
        return compton

    def setElasticModel(self):
        """
        setup parameters related to Elastic model
        """
        elastic = ElasticModel(prefix='elastic_')

        item = 'coherent_sct_amplitude'
        if item in self.parameter.keys():
            _set_parameter_hint(item, self.parameter[item], elastic)
        else:
            _set_parameter_hint(item, self.parameter_default[item], elastic)

        logger.debug(' ###### Started setting up parameters for elastic model. ######')

        # set constraints for the following global parameters
        elastic.set_param_hint('e_offset', expr='e_offset')
        elastic.set_param_hint('e_linear', expr='e_linear')
        elastic.set_param_hint('e_quadratic', expr='e_quadratic')
        elastic.set_param_hint('fwhm_offset', expr='fwhm_offset')
        elastic.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')
        elastic.set_param_hint('coherent_sct_energy', expr='coherent_sct_energy')
        logger.debug(' Finished setting up parameters for elastic model.')

        return elastic

    def model_spectrum(self):
        """
        Add all element peaks to the model.
        """
        incident_energy = self.incident_energy
        element_list = self.element_list
        parameter = self.parameter

        mod = self.setComptonModel() + self.setElasticModel()

        width_adjust = [item.split('-')[1] for item in list(parameter.keys()) if item.startswith('width')]
        pos_adjust = [item.split('-')[1] for item in list(parameter.keys()) if item.startswith('pos')]
        ratio_adjust = [item.split('-')[1] for item in list(parameter.keys()) if item.startswith('ratio')]

        ratio_set = []
        if parameter.has_key('set_branch_ratio'):
            ratio_set = parameter['set_branch_ratio'].keys()
            logger.info(' The branching ratio for those elements'
                        ' will be reset by users: {0}.'.format(ratio_adjust))
        else:
            logger.info(' No adjustment on branching ratio needs to be considered.')

        for ename in element_list:
            if ename in k_line:
                e = Element(ename)
                if e.cs(incident_energy)['ka1'] == 0:
                    logger.info(' {0} Ka emission line is not activated '
                                'at this energy {1}'.format(ename, incident_energy))
                    continue

                logger.debug(' --- Started building {0} peak. ---'.format(ename))

                for num, item in enumerate(e.emission_line.all[:4]):
                    line_name = item[0]
                    val = item[1]

                    if e.cs(incident_energy)[line_name] == 0:
                        continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')
                    gauss_mod.set_param_hint('e_offset', expr='e_offset')
                    gauss_mod.set_param_hint('e_linear', expr='e_linear')
                    gauss_mod.set_param_hint('e_quadratic', expr='e_quadratic')
                    gauss_mod.set_param_hint('fwhm_offset', expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')

                    if line_name == 'ka1':
                        gauss_mod.set_param_hint('area', value=100, vary=True, min=0)
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True, min=0,
                                                 expr=str(ename)+'_ka1_'+'area')
                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('delta_sigma', value=0, vary=False)
                    gauss_mod.set_param_hint('delta_center', value=0, vary=False)
                    ratio_v = e.cs(incident_energy)[line_name]/e.cs(incident_energy)['ka1']
                    gauss_mod.set_param_hint('ratio', value=ratio_v, vary=False)
                    gauss_mod.set_param_hint('ratio_adjust', value=0, vary=False)
                    logger.info(' {0} {1} peak is at energy {2} with'
                                ' branching ratio {3}.'. format(ename, line_name, val, ratio_v))

                    # position needs to be adjusted
                    if ename in pos_adjust:
                        pos_name = 'pos-'+ename+'-'+str(line_name)
                        if parameter.has_key(pos_name):
                            _set_parameter_hint('delta_center', parameter[pos_name],
                                                gauss_mod, log_option=True)

                    # width needs to be adjusted
                    if ename in width_adjust:
                        width_name = 'width-'+ename+'-'+str(line_name)
                        if parameter.has_key(width_name):
                            _set_parameter_hint('delta_sigma', parameter[width_name],
                                                gauss_mod, log_option=True)

                    # branching ratio needs to be adjusted
                    if ename in ratio_adjust:
                        ratio_name = 'ratio-'+ename+'-'+str(line_name)
                        if parameter.has_key(ratio_name):
                            #parameter[ratio_name]['value'] *= ratio_v
                            #parameter[ratio_name]['min'] *= ratio_v
                            #parameter[ratio_name]['max'] *= ratio_v
                            _set_parameter_hint('ratio_adjust', parameter[ratio_name],
                                                gauss_mod, log_option=True)

                    # fit branching ratio
                    #if ename in ratio_adjust:
                    #    if parameter['fit_branch_ratio'][ename].has_key(line_name.lower()):
                    #        ratio_change = parameter['fit_branch_ratio'][ename][line_name.lower()]
                    #        if ratio_change[0] == ratio_change[1]:
                    #            gauss_mod.set_param_hint('ratio', value=ratio_v*ratio_change[0], vary=False)
                    #            logger.warning(' Set branching ratio of {0} {1} as {2}.'.
                    #                           format(ename, line_name, ratio_v*ratio_change[0]))

                    #        else:
                    #            minr = min(ratio_change)
                    #            maxr = max(ratio_change)
                    #            gauss_mod.set_param_hint('ratio', value=ratio_v, vary=True,
                    #                                     min=ratio_v*minr,
                    #                                     max=ratio_v*maxr)
                    #            logger.warning(' Fit branching ratio of {0} {1}'
                    #                           ' within range {2}.'.format(ename, line_name,
                    #                                                       [minr*ratio_v,
                    #                                                        maxr*ratio_v]))

                    # set branching ratio
                    if ename in ratio_set:
                        if parameter['set_branch_ratio'][ename].has_key(line_name.lower()):
                            ratio_new = parameter['set_branch_ratio'][ename][line_name.lower()]
                            gauss_mod.set_param_hint('ratio', value=ratio_v*ratio_new, vary=False)
                            logger.warning(' Set branching ratio of {0} {1} as {2}.'.
                                           format(ename, line_name, ratio_v*ratio_new))

                    mod = mod + gauss_mod
                logger.debug(' Finished building element peak for {0}'.format(ename))

            elif ename in l_line:
                ename = ename[:-2]
                e = Element(ename)
                if e.cs(incident_energy)['la1'] == 0:
                    logger.info('{0} La1 emission line is not activated '
                                'at this energy {1}'.format(ename, incident_energy))
                    continue

                # L lines
                #gauss_mod = GaussModel_Llines(prefix=str(ename)+'_l_line_')

                #gauss_mod.set_param_hint('area', value=100, vary=True, min=0)
                #gauss_mod.set_param_hint('fwhm_offset', value=0.1, vary=True, expr='fwhm_offset')
                #gauss_mod.set_param_hint('fwhm_fanoprime', value=0.1, vary=True, expr='fwhm_fanoprime')

                for num, item in enumerate(e.emission_line.all[4:-4]):

                    line_name = item[0]
                    val = item[1]

                    if e.cs(incident_energy)[line_name] == 0:
                        continue

                    gauss_mod = GaussModel_xrf(prefix=str(ename)+'_'+str(line_name)+'_')

                    gauss_mod.set_param_hint('fwhm_offset', expr='fwhm_offset')
                    gauss_mod.set_param_hint('fwhm_fanoprime', expr='fwhm_fanoprime')

                    if line_name == 'la1':
                        gauss_mod.set_param_hint('area', value=100, vary=True)
                                             #expr=gauss_mod.prefix+'ratio_val * '+str(ename)+'_la1_'+'area')
                    else:
                        gauss_mod.set_param_hint('area', value=100, vary=True,
                                                 expr=str(ename)+'_la1_'+'area')

                    gauss_mod.set_param_hint('center', value=val, vary=False)
                    gauss_mod.set_param_hint('sigma', value=1, vary=False)
                    gauss_mod.set_param_hint('ratio',
                                             value=e.cs(incident_energy)[line_name]/e.cs(incident_energy)['la1'],
                                             vary=False)

                    mod = mod + gauss_mod

        self.mod = mod
        return

    def model_fit(self, x, y, w=None, method='leastsq', **kws):
        """
        Parameters
        ----------
        x : array
            independent variable
        y : array
            intensity
        w : array, optional
            weight for fitting
        method : str
            default as leastsq
        kws : dict
            fitting criteria, such as max number of iteration

        Returns
        -------
        obj
            saving all the fitting results
        """

        self.model_spectrum()

        pars = self.mod.make_params()
        result = self.mod.fit(y, pars, x=x, weights=w,
                              method=method, fit_kws=kws)
        return result
