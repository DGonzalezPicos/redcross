import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.constants as const
from astropy import units as u, coordinates as coord
from scipy.interpolate import interp1d
class Planet:
    
    def __init__(self, file=None, name=None, **header):
        # if name == 'WASP189':
        #     self.P = 2.7240338 # d
        #     self.a = 0.0497 # AU
        #     self.i = 84.32 # deg
        #     self.Tc_jd = 2456706.4558
        #     self.T_14 = 0.1813 * 24. # d
        #     self.v_sys = -20.82 #km/s
        #     self.RA_DEG = 225.68695
        #     self.DEC_DEG = -3.0313833

        if not file is None:
            pvalues = np.loadtxt(file)
            keys = ['P','a', 'i', 'v_sys', 'Tc_jd', 'T_14']
            for key, value in zip(keys, pvalues):
                setattr(self, key, value)
                
        for key in header:
            setattr(self, key, header[key])
            
        if hasattr(self, 'Tc_jd'): self.Tc = Time(self.Tc_jd, format='jd',scale='tdb') 
        if hasattr(self, 'T_14'): self.T_14 /= 24. # from hours to days

        if hasattr(self, 'a'):
            self.v_orb = (2*np.pi*self.a*u.AU / (self.P*u.d)).to(u.km/u.s)
            self.Kp = (self.v_orb * np.sin(np.radians(self.i))).value
            
        self.frame = 'telluric' # default

    @property
    def BJD(self, location='orm'):
        '''convert MJD to BJD'''
        from astropy.coordinates import SkyCoord, EarthLocation
        if location == 'orm':
            self.location = EarthLocation.of_site('Roque de los Muchachos')  # download site data
        else:
            try:
                self.location = EarthLocation.of_site(location) 
            except:
                print('Please provide a valid astropy EarthLocation quantity!')
            
        target = SkyCoord(self.RA_DEG,
                                   self.DEC_DEG,unit=(u.deg, u.deg), 
                                   frame='icrs')
        
        #Convert MJD to BJD to account for light travel time. Adopted from Astropy manual.
#        print(self.location.geodetic)
        t = Time(self.MJD, format='mjd',scale='tdb',location=self.location) 
        ltt_bary = t.light_travel_time(target)
        return t.tdb + ltt_bary # = BJD  
    
    @property
    def phase(self):
        return ((self.BJD-self.Tc).value % self.P) / self.P
    
    @property
    def RV(self):
        #Derive the instantaneous radial velocity at which the planet is expected to be.
        RV_planet = np.sin(2*np.pi*self.phase)*self.Kp
        if self.frame == 'stellar':
            rvel = 0.0
        elif self.frame == 'telluric':
            rvel = ((self.v_sys*u.km/u.s)-self.BERV*u.km/u.s).value
            
        elif self.frame == 'planet':
            return np.zeros_like(self.BERV)
            
            
        return (rvel + RV_planet)
    
    def interpolate(self, newX):
        # Update planet header with the interpolated vectors covering the gap
    
        newPhase = np.linspace(self.phase.min(), self.phase.max(), newX.size)
        newMJD = interp1d(self.phase, self.MJD)(newPhase)
        newBERV = interp1d(self.phase, self.BERV)(newPhase)
       
        self.MJD = newMJD
        self.BERV = newBERV
        
        return self
    
    def mask_eclipse(self, invert_mask=False, return_mask=False, debug=False):
        '''given the duration of eclipse `t_14` in days
       return the PLANET with the frames masked'''
      
        shape_in = self.RV.size
        phase = self.phase
        phase_14 = 0.5 * ((self.T_14) % self.P) / self.P
        
        mask = np.abs(phase - 0.50) < phase_14 # frames IN-eclipse
#        mask = (phase > (0.50 - (0.5*phase_14)))*(phase < (0.50 + (0.5*phase_14)))
        if invert_mask:
            mask = ~mask
    
        

            
        if return_mask:
            return mask
        else:
            # Update planet header with the MASKED vectors
            for key in ['MJD','BERV','airmass']:
                # self.header[item] = self.header[item][~mask]
                setattr(self, key, getattr(self, key)[~mask])
                
            if debug:
                print('Original self.shape = {:}'.format(shape_in))
                print('After ECLIPSE masking = {:}'.format(self.RV.size))
            return self
    
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def save(self, outname):
        np.save(outname, self.__dict__) 
        print('{:} saved...'.format(outname))
        return None
    
    def load(self, path):
        print('Loading Datacube from...', path)
        d = np.load(path, allow_pickle=True).tolist()
        for key in d.keys():
            setattr(self, key, d[key])
        return self    
    
    def trail(self, ax, frame='telluric', top=20, bottom=40, **kwargs):
        self.frame = frame
        ax.plot(self.RV[-top:], self.phase[-top:], '--r', **kwargs)
        ax.plot(self.RV[:bottom], self.phase[:bottom], '--r', **kwargs)
        return None
        
