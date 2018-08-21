Copyright (c) 2017, Universtitat Jaume I (UJI)
The data is licensed under CC Attribution 4.0 International (CC BY 4.0).
The script files are provided under MIT license / X11 license.
This documentation is licensed under CC0 license.

WiFi measurements database for UJI's library and supporting material.
Version 0.4 24-11-2017

Directory content:

./      ->      Common utility functions and sample usage Matlab scripts.
db/     ->      Database measurements. Within it, each subfolder contains the measurements
				of each dataset that belongs to a collection month. Each dataset is represented
				by four files: the RSS, the time, the coordinates, and the identifiers files.
files/  ->      Matlab functions for loading the measurements.
ids/    ->      Matlab functions for filtering loaded samples.
ips/	->      Matlab functions provinding sample IPS methods.
shelves/ ->     Bookshelves coordinates and related Matlab scripts.


Citation request: 

G.M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E.S. Lohan, J. Huerta, A. Cramariuc, "Long-Term
Wi-Fi fingerprinting dataset and supporting material", Zenodo repository, DOI 10.5281/zenodo.1066041.

gmendoza@uji.es
philipp.richter@tut.fi
jtorres@uji.es
elena-simona.lohan@tut.fi
huerta@uji.es
