import numpy as np

hed_data = np.loadtxt("HED_raw.txt").T
led_data = np.loadtxt("LED_raw.txt").T



def get_event_time(mus, ttk):
    return mus + ttk * 1 / 36


hed_compton_photon_indices = []

mus = -np.inf
ttk = 0
current_event_time = get_event_time(mus, ttk)

for i, event in enumerate(hed_data.T):
    next_mus = event[0]
    next_ttk = event[1]
    next_event_time = get_event_time(next_mus, next_ttk)
    # if next_event_time - current_event_time > 0.200:
    if next_event_time - current_event_time > 0.200:
        hed_compton_photon_indices.append(i)
    current_event_time = next_event_time

cleaned_hed_ttes = hed_data[0][hed_compton_photon_indices]
cleaned_hed_ttes *= 1e-6


led_compton_photon_indices = np.where(led_data[5] == 0)[0]  # exclude photons that are double counts
cleaned_led_ttes = led_data[0]
cleaned_led_ttes = np.unique(cleaned_led_ttes)
cleaned_led_ttes *= 1e-6

np.savetxt("ASIM_CLEANED_HED.txt", cleaned_hed_ttes)
np.savetxt("ASIM_CLEANED_LED.txt", cleaned_led_ttes)

cleaned_led_ttes = cleaned_led_ttes[np.where(np.logical_and(cleaned_led_ttes >= 80e-3, cleaned_led_ttes <= 100e-3))[0]]
cleaned_hed_ttes = cleaned_hed_ttes[np.where(np.logical_and(cleaned_hed_ttes >= 80e-3, cleaned_hed_ttes <= 100e-3))[0]]
print(len(cleaned_hed_ttes))
print(len(cleaned_led_ttes))
