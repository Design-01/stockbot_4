from dataclasses import dataclass
import pandas as pd
import random

@dataclass
class RandDataStore:
    """Stores random data for testing."""
    date   : pd.DatetimeIndex 
    open   : tuple = ()
    high   : tuple = ()
    low    : tuple = ()
    close  : tuple = ()
    volume : tuple = ()

@dataclass
class RandomOHLCV:
    """Creates random data for testing."""
    open_rng  : tuple = (0.4,0.4)         # Range % change (min, max) from one bar to the next #* making min more neg than max gives a nore bias
    close_rng : tuple = (0.4,0.4)        # Range % change(min, max)  from one bar to the next
    start     : str   = '2022' # start date as string ge '2022'
    periods   : int   = 50     # length of the data
    freq      : str   ='5 min' # eg 'H', '4H', '5 min', 'D'
    open_val  : float = 100    # very first open price
    head_max  : int   = 5      # max % change  of last close added to bar to get high
    tail_max  : int   = 5      # max % change of last close subtracted from bar to get low
    vol_rng   : tuple = (-50, 60 )# Range % change (min, max) from one bar to the next
    vol_start : int   = 500    # starting volume
    volatility_rng: tuple = (0, 0) # Range % change (min, max) from one bar to the next
    volatility_freq: int = 0       # frequency of volatility change. 0 = every bar, 1 = every other bar, 2 = every 3rd bar etc
    volatility_dur : int = 0       # duration of volatility change. 0 = 1 bar, 1 = 2 bars, 2 = 3 bars etc

    
    def get_volatility(self, i, duration):
        """ returns a random volatility value considering frequency and duration """
        if self.volatility_freq > 0 and i % self.volatility_freq == 0:
            duration += 1
            return 1 + random.uniform(self.volatility_rng[0], self.volatility_rng[1]), duration
        
        if 0 < duration <= self.volatility_dur:
            duration += 1
            return 1 + random.uniform(self.volatility_rng[0], self.volatility_rng[1]), duration
        
        if duration > self.volatility_dur:
            duration = 0
        
        return 1, duration

    def get_multiplying_factor(self, freq):
        """Calculate the multiplying factor based on the frequency."""
        freq_map = {
            '1 T': 1,
            '5 T': 1,
            '15 T': 1,
            '30 T': 2,
            '60 T': 3,
            '1 H': 3,
            '4 H': 5,
            '1 D': 30
        }
        return freq_map.get(freq, 1)
    
    def get_rand_val(self, val, rng, minVal, maxVal, div=1, volotility=1):
        """ retruns a random value based on the val, rng, minVal, maxVal, div and volotility. 
        if the value is less than minVal then the close_rng is used to bring the value back up to minVal. 
        vice versa for maxVal."""
        if   val < minVal: close_rng = (max(0, rng[0]), rng[1])
        elif val > maxVal: close_rng = (rng[0], min(0, rng[1]))
        else: close_rng = rng
        return val * (1 + ( random.uniform(close_rng[0], close_rng[1]) / div ) * volotility)
    
    def convert_to_pd_freq_format(self, freq):
        """string must separate the number and the char with a space. eg '5 min'"""
        if freq.isdigit():
            return freq + 'D'
        parts = freq.split(' ')
        if len(parts) == 2:
            numb = int(parts[0])
            char = 'min' if 'min' in parts[1].lower() else parts[1][0].upper()
            return f'{numb} {char}'
        return freq
    
    def get_rand_volume(self, prev_volume: float, vol_factor: float) -> int:
        """
        Generate random volume with natural variation and occasional spikes.
        """
        # Larger random component (30-170% of previous volume)
        random_change = random.uniform(0.3, 1.7)
        new_volume = prev_volume * random_change
        
        # Occasional larger spikes (5% chance)
        if random.random() < 0.05:
            new_volume *= random.uniform(1.5, 3.0)
        
        # Base constraints
        vol_min = 5 * vol_factor
        vol_max = 100 * vol_factor
        
        return int(min(max(vol_min, new_volume), vol_max))

    def __post_init__(self) -> None:
        self.freq = self.convert_to_pd_freq_format(self.freq)
        factor    = self.get_multiplying_factor(self.freq) 
        self.open_rng = (self.open_rng[0] * factor, self.open_rng[1] * factor)
        self.close_rng = (self.close_rng[0] * factor, self.close_rng[1] * factor)
        self.volatility_rng = (self.volatility_rng[0] * factor, self.volatility_rng[1] * factor)
        self.head_max *= factor
        self.tail_max *= factor

        self.data = RandDataStore(pd.date_range(start=self.start, periods=self.periods, freq=self.freq))
        o, c, v = self.open_val, self.open_val, self.vol_start * factor 
        duration = 0
        vola = 1
        # vol_factor = factor * 40 if self.freq == '1 D' else factor * 10
        vol_factor = factor 


        base_volume = self.vol_start * factor
        v = base_volume  # Start at base volume

        for i in range(len(self.data.date)):
            vola, duration = self.get_volatility(i, duration)

            o = round(self.get_rand_val(c, self.open_rng,  minVal=20,  maxVal=500, div=100),  2)
            c = round(self.get_rand_val(o, self.close_rng, minVal=20,  maxVal=500, div=100, volotility=vola), 2)
            v = self.get_rand_volume(v, vol_factor)
            c = max(5, min(c, 600)) # stop close going below 1

            self.data.open   += (o,)
            self.data.close  += (c,)
            self.data.volume += (v,)

        self.data.high   = tuple(round(max(o, c) + (o *  random.uniform(0, self.head_max)/100), 2) for o, c in  zip(self.data.open, self.data.close))
        self.data.low    = tuple(round(min(o, c) - (o *  random.uniform(0, self.tail_max)/100), 2) for o, c in  zip(self.data.open, self.data.close))

    def get_dataframe(self):
        return pd.DataFrame(self.data.__dict__).set_index('date', drop=True)
