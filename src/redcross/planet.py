import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.constants as const
from astropy import units as u, coordinates as coord
from scipy.interpolate import interp1d
class Planet:
    
    def __init__(self, name=None, file=None, **header):
        if name == 'WASP189':
            self.P = 2.7240338 # d
            self.a = 0.0497 # AU
            self.i = 84.32 # deg
            self.Tc_jd = 2456706.4558
            self.T_14 = 0.1813 * 24. # d
            self.v_sys = -20.82 #km/s
            self.RA_DEG = 225.68695
            self.DEC_DEG = -3.031383

        elif file != None:
            pvalues = np.loadtxt(name)
            keys = ['P','a', 'i', 'v_sys', 'Tc_jd', 'T_14']
            for key, value in zip(keys, pvalues):
                setattr(self, key, value)
                
        for key in header:
            setattr(self, key, header[key])
            
        self.Tc = Time(self.Tc_jd, format='jd',scale='tdb') 
        if hasattr(self, 'T_14'): self.T_14 /= 24. # from hours to days

        self.v_orb = (2*np.pi*self.a*u.AU / (self.P*u.d)).to(u.km/u.s)
        self.Kp = self.v_orb / np.sin(np.radians(self.i))

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
        rvel = (self.v_sys*u.km/u.s)-self.BERV*u.km/u.s

        return rvel + (np.sin(2*np.pi*self.phase)*self.v_orb*np.sin(np.radians(self.i)))
    
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
        phase_14 = ((self.T_14) % self.P) / self.P
        
        mask = np.abs(phase - 0.50) < (phase_14/2.) # frames IN-eclipse
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

    


if __name__ == '__main__':     
    from datacube import Datacube, CCF, Template
    import glob
    
    
    template_file = glob.glob('data/*.fits')[0]
    template = Template(template_file).check_data()
    dc = Datacube().load('dc_raw-nObs297.npy', 'full').crop([6450,6550])
    planet = Planet('wasp189', **dc.header)

    def vactoair(wavelength):
        """VACUUM to AIR conversion as actually implemented by wcslib.
        Input wavelength with astropy.unit
        """
        wave = wavelength.to(u.m).value
        n = 1.0
        for k in range(4):
            s = (n/wave)**2
            n = 2.554e8 / (0.41e14 - s)
            n += 294.981e8 / (1.46e14 - s)
            n += 1.000064328
        return (wavelength / n).value
    
    def airtovac(wlA):
        #Convert wavelengths (AA) in air to wavelengths (AA) in vaccuum (empirical).
        s = 1e4 / wlA
        n = 1 + (0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) +
        0.0001599740894897 / (38.92568793293 - s**2))
        return(wlA*n)
    
    fig, ax = plt.subplots(3,1, figsize=(10,9))
    ax[0].plot(template.wlt, template.flux, label='original')
    ax[0].plot(vactoair(template.wlt*u.AA), template.flux, label='airtovac')
    
    shift = (vactoair(template.wlt*u.AA) - template.wlt) / template.wlt
    ax[1].plot(template.wlt,shift, '-g')
    ax[1].set(xlabel='Wavelength (A)', ylabel='Relative wavelength shift')
    
    ax[0].grid()
    ax[0].set_xlim(5000,5010)
    ax[0].set(xlabel='Wavelength (A)', ylabel='Flux')
    ax[0].legend()
    
    
    rv_shift = (np.mean(shift)* const.c).to('km/s').value
    print('RV-shift from VAC-to-AIR transformation {:.2f} km/s'.format(rv_shift))
    ax[2].plot(planet.RV, planet.phase, label='original')
    ax[2].plot(planet.RV.value+rv_shift, planet.phase, label='shifted {:.2f} km/s'.format(rv_shift))
    ax[2].legend()
    ax[2].set(xlabel='RV (km/s)', ylabel='Phase')
    plt.show()
