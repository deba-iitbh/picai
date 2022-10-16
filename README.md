# The PI-CAI Challenge: Public Training and Development Dataset

## Reference
Please cite the following article, if you are using the PI-CAI: Public Training and Development Dataset:
	
	A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6522364

	@ARTICLE{PICAI_BIAS,
      	author = {Anindo Saha, Jasper J. Twilt, Joeran S. Bosma, Bram van Ginneken, Derya Yakar, Mattijs Elschot, Jeroen Veltman, Jurgen Fütterer, Maarten de Rooij, Henkjan Huisman},
      	title  = {{Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge}}, 
      	year   = {2022},
      	doi    = {10.5281/zenodo.6522364}
    }


## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)


## Dataset Characteristics
This dataset contains 1500 anonymized prostate biparametric MRI scans from 1476 patients, acquired between 2012-2021, at three centers (Radboud University Medical Center, University Medical Center Groningen, Ziekenhuis Groep Twente) based in The Netherlands. 


|                                |                 |
|--------------------------------|-----------------|
| Number of sites                | 11              |
| Number of MRI scanners         | 5 S, 2 P        |
| Number of patients             | 1476            |
| Number of cases                | 1500            |
| — Benign or indolent PCa       | 1075            |
| — csPCa (ISUP ≥ 2)             | 425             |
| Median age (years)             | 66 (IQR: 61–70) |
| Median PSA (ng/mL)             | 8.5 (IQR: 6–13) |
| Median prostate volume (mL)    | 57 (IQR: 40–80) |
| Number of positive MRI lesions | 1087            |
| — PI-RADS 3                    | 246 (23%)       |
| — PI-RADS 4                    | 438 (40%)       |
| — PI-RADS 5                    | 403 (37%)       |
| Number of ISUP-based lesions   | 776             |
| — ISUP 1                       | 311 (40%)       |
| — ISUP 2                       | 260 (34%)       |
| — ISUP 3                       | 109 (14%)       |
| — ISUP 4                       | 41 (5%)         |
| — ISUP 5                       | 55 (7%)         |

Abbreviations:
- S: Siemens Healthineers
- P: Philips Medical Systems
- PCa: prostate cancer
- csPCa: clinically significant prostate cancer

## Imaging Files

Imaging sequences are mapped to filenames in the following way:

* Axial T2-weighted imaging (T2W): `[patient_id]_[study_id]_t2w.mha`
* Axial high b-value (≥ 1000 s/mm2) diffusion-weighted imaging (HBV): `[patient_id]_[study_id]_hbv.mha`
* Axial apparent diffusion coefficient maps (ADC): `[patient_id]_[study_id]_adc.mha`
* Sagittal T2-weighted imaging: `[patient_id]_[study_id]_sag.mha`
* Coronal T2-weighted imaging: `[patient_id]_[study_id]_cor.mha`

Every patient case will at least have three imaging sequences: axial T2W, axial HBV and axial ADC scans (i.e., files ending in `_t2w.mha`, `_hbv.mha`, `_adc.mha`). Additionally, they can have either, both or none of the sagittal and coronal T2W scans (i.e., files ending in `_sag.mha`, `_cor.mha`).


## Folder Structure

```
images  (root folder with all patients, and in turn, all 1500 studies)
├── ...
├── 10417  (patient-level folder, including all studies for a given patient)
	├── 10417_1000424_t2w.mha  (axial T2W imaging for study 1000424)
	├── 10417_1000424_adc.mha  (axial ADC imaging for study 1000424)
	├── ...
	├── 10417_1000425_t2w.mha  (axial T2W imaging for study 1000425)
	├── 10417_1000425_adc.mha  (axial ADC imaging for study 1000425)
	├── ...
├── ...
```


## Annotations for Dataset
See [https://github.com/DIAGNijmegen/picai_labels](https://github.com/DIAGNijmegen/picai_labels).


## Managed By
Diagnostic Image Analysis Group,
Radboud University Medical Center,
Nijmegen, The Netherlands

## Contact Information
- Anindo Saha: Anindya.Shaha@radboudumc.nl
- Jasper Twilt: Jasper.Twilt@radboudumc.nl
- Joeran Bosma: Joeran.Bosma@radboudumc.nl
- Henkjan Huisman: Henkjan.Huisman@radboudumc.nl
