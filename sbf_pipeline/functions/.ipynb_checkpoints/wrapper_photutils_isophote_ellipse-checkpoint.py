from photutils.isophote import Ellipse

# copied imports from the original method 
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from photutils.isophote.fitter import (DEFAULT_CONVERGENCE, DEFAULT_FFLAG,
                                       DEFAULT_MAXGERR, DEFAULT_MAXIT,
                                       DEFAULT_MINIT, CentralEllipseFitter,
                                       EllipseFitter)
from photutils.isophote.geometry import EllipseGeometry
from photutils.isophote.integrator import BILINEAR
from photutils.isophote.isophote import Isophote, IsophoteList
from photutils.isophote.sample import CentralEllipseSample, EllipseSample
from photutils.utils._deprecation import (deprecated_positional_kwargs,
                                          deprecated_renamed_argument)



class WrappedEllipse(Ellipse):    
    # This is the function that is modified using this wrapper. The 
    # parameter 'inside_non_fixed' is set to True to use the 
    # alteration and works as before when kept on False
    
    def fit_image(self, sma0=None, minsma=0.0, maxsma=None, step=0.1,
                  conver=DEFAULT_CONVERGENCE, minit=DEFAULT_MINIT,
                  maxit=DEFAULT_MAXIT, fflag=DEFAULT_FFLAG,
                  maxgerr=DEFAULT_MAXGERR, sclip=3.0, n_clip=0,
                  integrmode=BILINEAR, linear=None, maxrit=None,
                  fix_center=False, fix_pa=False, fix_eps=False,
                  inside_non_fixed=False):
        # This parameter list is quite large and should in principle be
        # simplified by redistributing these controls to somewhere else.
        # We keep this design though because it better mimics the flat
        # architecture used in the original STSDAS task `ellipse`.
        """
        Fit multiple isophotes to the image array.

        This method loops over each value of the semimajor axis (sma)
        length (constructed from the input parameters), fitting a single
        isophote at each sma. The entire set of isophotes is returned in
        an `~photutils.isophote.IsophoteList` instance.

        Note that the fix_XXX parameters act in unison. Meaning,
        if one of them is set via this call, the others will
        assume their default (False) values. This effectively
        overrides any settings that are present in the internal
        `~photutils.isophote.EllipseGeometry` instance that is
        carried along as a property of this class. If an instance of
        `~photutils.isophote.EllipseGeometry` was passed to this class'
        constructor, that instance will be effectively overridden by the
        fix_XXX parameters in this call.

        Parameters
        ----------
        sma0 : float, optional
            The starting value for the semimajor axis length (pixels).
            This value must not be the minimum or maximum semimajor
            axis length, but something in between. The algorithm can't
            start from the very center of the galaxy image because
            the modelling of elliptical isophotes on that region is
            poor and it will diverge very easily if not tied to other
            previously fit isophotes. It can't start from the maximum
            value either because the maximum is not known beforehand,
            depending on signal-to-noise. The ``sma0`` value should be
            selected such that the corresponding isophote has a good
            signal-to-noise ratio and a clearly defined geometry. If set
            to `None` (the default), one of two actions will be taken:
            if a `~photutils.isophote.EllipseGeometry` instance was
            input to the `~photutils.isophote.Ellipse` constructor, its
            ``sma`` value will be used. Otherwise, a default value of
            10. will be used.

        minsma : float, optional
            The minimum value for the semimajor axis length (pixels).
            The default is 0.

        maxsma : float or `None`, optional
            The maximum value for the semimajor axis length (pixels).
            When set to `None` (default), the algorithm will increase
            the semimajor axis until one of several conditions will
            cause it to stop and revert to fit ellipses with sma <
            ``sma0``.

        step : float, optional
            The step value used to grow/shrink the semimajor axis
            length (pixels if ``linear=True``, or a relative value if
            ``linear=False``). See the ``linear`` parameter. The default
            is 0.1.

        conver : float, optional
            The main convergence criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than ``conver`` times the harmonic fit rms. The
            default is 0.05.

        minit : int, optional
            The minimum number of iterations to perform. A minimum of
            10 (the default) iterations guarantees that, on average, 2
            iterations will be available for fitting each independent
            parameter (the four harmonic amplitudes and the intensity
            level). For the first isophote, the minimum number of
            iterations is 2 * ``minit`` to ensure that, even departing
            from not-so-good initial values, the algorithm has a better
            chance to converge to a sensible solution.

        maxit : int, optional
            The maximum number of iterations to perform. The default is
            50.

        fflag : float, optional
            The acceptable fraction of flagged data points in the
            sample. If the actual fraction of valid data points is
            smaller than this, the iterations will stop and the current
            `~photutils.isophote.Isophote` will be returned. Flagged
            data points are points that either lie outside the image
            frame, are masked, or were rejected by sigma-clipping. The
            default is 0.7.

        maxgerr : float, optional
            The maximum acceptable relative error in the local
            radial intensity gradient. This is the main control
            for preventing ellipses to grow to regions of too
            low signal-to-noise ratio. It specifies the maximum
            acceptable relative error in the local radial
            intensity gradient. `Busko (1996; ASPC 101, 139)
            <https://ui.adsabs.harvard.edu/abs/1996ASPC..101..139B/abstr
            act>`_ showed that the fitting precision relates to that
            relative error. The usual behavior of the gradient relative
            error is to increase with semimajor axis, being larger in
            outer, fainter regions of a galaxy image. In the current
            implementation, the ``maxgerr`` criterion is triggered only
            when two consecutive isophotes exceed the value specified by
            the parameter. This prevents premature stopping caused by
            contamination such as stars and HII regions.

            A number of actions may happen when the gradient error
            exceeds ``maxgerr`` (or becomes non-significant and is
            set to `None`). If the maximum semimajor axis specified
            by ``maxsma`` is set to `None`, semimajor axis growth is
            stopped and the algorithm proceeds inwards to the galaxy
            center. If ``maxsma`` is set to some finite value, and this
            value is larger than the current semimajor axis length, the
            algorithm enters non-iterative mode and proceeds outwards
            until reaching ``maxsma``. The default is 0.5.

        sclip : float, optional
            The sigma-clip sigma value. The default is 3.0.

        n_clip : int, optional
            The number of sigma-clip iterations. The default is 0, which
            means sigma-clipping is skipped.

            .. deprecated:: 3.0
                The ``nclip`` keyword is deprecated. Use ``n_clip``
                instead.

        integrmode : {'bilinear', 'nearest_neighbor', 'mean', 'median'}, \
                optional
            The area integration mode. The default is 'bilinear'.

        linear : bool, optional
            The semimajor axis growing/shrinking mode. If `False`
            (default), the geometric growing mode is chosen, thus the
            semimajor axis length is increased by a factor of (1.
            + ``step``), and the process is repeated until either
            the semimajor axis value reaches the value of parameter
            ``maxsma``, or the last fitted ellipse has more than a given
            fraction of its sampled points flagged out (see ``fflag``).
            The process then resumes from the first fitted ellipse (at
            ``sma0``) inwards, in steps of (1./(1. + ``step``)), until
            the semimajor axis length reaches the value ``minsma``. In
            case of linear growing, the increment or decrement value
            is given directly by ``step`` in pixels. If ``maxsma`` is
            set to `None`, the semimajor axis will grow until a low
            signal-to-noise criterion is met. See ``maxgerr``.

        maxrit : float or `None`, optional
            The maximum value of semimajor axis to perform an actual
            fit. Whenever the current semimajor axis length is larger
            than ``maxrit``, the isophotes will be extracted using the
            current geometry, without being fitted. This non-iterative
            mode may be useful for sampling regions of very low surface
            brightness, where the algorithm may become unstable
            and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically
            whenever the ellipticity exceeds 1.0 or the ellipse center
            crosses the image boundaries. If `None` (default), then no
            maximum value is used.

        fix_center : bool, optional
            Keep center of ellipse fixed during fit? The default is
            False.

        fix_pa : bool, optional
            Keep position angle of semi-major axis of ellipse fixed
            during fit? The default is False.

        fix_eps : bool, optional
            Keep ellipticity of ellipse fixed during fit? The default is
            False.

        Returns
        -------
        result : `~photutils.isophote.IsophoteList` instance
            A list-like object of `~photutils.isophote.Isophote`
            instances, sorted by increasing semimajor axis length.
        """
        # multiple fitted isophotes will be stored here
        isophote_list = []

        # get starting sma from appropriate source: keyword parameter,
        # internal EllipseGeometry instance, or fixed default value.
        if not sma0:
            sma = self._geometry.sma if self._geometry else 10.0
        else:
            sma = sma0

        # Override geometry instance with parameters set at the call.
        if isinstance(linear, bool):
            self._geometry.linear_growth = linear
        else:
            linear = self._geometry.linear_growth
        if fix_center and fix_pa and fix_eps:
            msg = ': Everything is fixed. Fit not possible.'
            warnings.warn(msg, AstropyUserWarning)
            return IsophoteList([])
        if fix_center or fix_pa or fix_eps:
            # Note that this overrides the geometry instance for good.
            self._geometry.fix = np.array([fix_center, fix_center, fix_pa,
                                           fix_eps])

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        noiter = False
        first_isophote = True
        while True:
            # first isophote runs longer
            minit_a = 2 * minit if first_isophote else minit
            first_isophote = False

            isophote = self.fit_isophote(sma, step=step, conver=conver,
                                         minit=minit_a, maxit=maxit,
                                         fflag=fflag, maxgerr=maxgerr,
                                         sclip=sclip, n_clip=n_clip,
                                         integrmode=integrmode,
                                         linear=linear, maxrit=maxrit,
                                         noniterate=noiter,
                                         isophote_list=isophote_list)

            # check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == 1:
                # in case the fit failed right at the outset, return an
                # empty list. This is the usual case when the user
                # provides initial guesses that are too way off to enable
                # the fitting algorithm to find any meaningful solution.

                if len(isophote_list) == 1:
                    msg = 'No meaningful fit was possible.'
                    warnings.warn(msg, AstropyUserWarning)
                    return IsophoteList([])

                self._fix_last_isophote(isophote_list, -1)

                # get last isophote from the actual list, since the last
                # `isophote` instance in this context may no longer be OK.
                isophote = isophote_list[-1]

                # if two consecutive isophotes failed to fit,
                # shut off iterative mode. Or, bail out and
                # change to go inwards.
                if (len(isophote_list) > 2
                    and ((isophote.stop_code == 5
                          and isophote_list[-2].stop_code == 5)
                         or isophote.stop_code == 1)):
                    if maxsma and maxsma > isophote.sma:
                        # if a maximum sma value was provided by
                        # user, and the current sma is smaller than
                        # maxsma, keep growing sma in non-iterative
                        # mode until reaching it.
                        noiter = True
                    else:
                        # if no maximum sma, stop growing and change
                        # to go inwards.
                        break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

            # update sma. If exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry.update_sma(step)
            if maxsma and sma >= maxsma:
                break

        # reset sma so as to go inwards.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)

        # OWN CODE ALTERATION
        if inside_non_fixed:
            print("inside_non_fixed modification used")
            self._geometry.fix = np.array([False, False, fix_pa, fix_eps])
            isophote_list[-1].sample.geometry.fix = np.array([False, False, fix_pa, fix_eps])
        
        # now, go from initial sma inwards towards center.
        while True:
            isophote = self.fit_isophote(sma, step=step, conver=conver,
                                         minit=minit, maxit=maxit,
                                         fflag=fflag, maxgerr=maxgerr,
                                         sclip=sclip, n_clip=n_clip,
                                         integrmode=integrmode,
                                         linear=linear, maxrit=maxrit,
                                         going_inwards=True,
                                         isophote_list=isophote_list)

            # if abnormal condition, fix isophote but keep going.
            if isophote.stop_code < 0:
                self._fix_last_isophote(isophote_list, 0)

            # but if we get an error from the scipy fitter, bail out
            # immediately. This usually happens at very small radii
            # when the number of data points is too small.
            if isophote.stop_code == 3:
                break

            # reset variable from the actual list, since the last
            # `isophote` instance may no longer be OK.
            isophote = isophote_list[-1]

            # figure out next sma; if exceeded user-defined
            # minimum, or too small, bail out from this loop
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.5):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            # isophote is appended to isophote_list
            _ = self.fit_isophote(0.0, isophote_list=isophote_list)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return IsophoteList(isophote_list)
