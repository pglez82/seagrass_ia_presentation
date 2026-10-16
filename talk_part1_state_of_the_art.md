# State of the Art: Artificial Intelligence Applied to the Mapping and Monitoring of Seagrass Meadows
### Preparation Guide & Literature Review for Part 1 of the Talk
**Authors of the talk:** Pablo González (AIC, Universidad de Oviedo) & Fernando García-González (IEO-CSIC)  
**Context:** CartoGrass / Marine Habitat Mapping Congress (Spain)

---

## 1. Executive Summary & Narrative Arc for Part 1

The objective of Part 1 is to take the audience on a clear methodological journey:
1. **The Ecological & Practical Challenge:** Why seagrasses matter (*Posidonia oceanica*, *Cymodocea nodosa*, *Zostera* spp.), their role as blue carbon sinks and coastal protectors, and why traditional in situ monitoring (diver transects, grab sampling) cannot scale to the requirements of the EU Habitats Directive, the Marine Strategy Framework Directive (MSFD), and national initiatives like the *Atlas de las Praderas de Fanerógamas Marinas de España*.
2. **Multi-Source Data Ecosystem:** The complementary roles of Satellite Remote Sensing (regional/temporal), UAVs/Drones (ultra-high spatial resolution), Underwater Imaging (diver/ROV/AUV ground truthing), and Acoustic Sensors (MBES, SSS for depth limits and turbid waters).
3. **Phase I — Classical Pixel-Based Paradigms:** Physics-based water column corrections (Lyzenga, Stumpf) coupled with pixel-wise classifiers (Maximum Likelihood, SVM, Random Forest). Why this paradigm hit a plateau (salt-and-pepper noise, mixed pixels, inability to grasp meadow morphology, leaf density, or seabed context).
4. **Phase II — Deep Learning & Spatial Context:** The shift to patch-based CNNs and semantic segmentation (U-Net, DeepLabv3+, SegNet, HRNet, Transformers). How spatial textures, meadow patchiness, and edge continuity are captured across satellites, drones, and acoustic backscatter.
5. **Phase III — Foundation Models & Self-Supervised Learning (SSL):** Addressing the acute bottleneck of benthic ground-truth scarcity. Masked Autoencoders (MAE), contrastive learning, geospatial foundation models (Prithvi, Clay, SatMAE), biological/underwater vision-language models (BioCLIP, AquaticCLIP), and promptable segmentation (SAM / MarineSAM).
6. **The Natural Bridge to Part 2:** How moving from isolated pixels to deep spatial and foundation models interacts with real-world **Dataset Shift** (temporal, environmental, optical, and sensor changes).

---

## 2. Multi-Source Remote Sensing Platforms: Strengths & Complementarity

| Platform / Sensor | Spatial Resolution | Temporal Revisit | Depth Penetration | Primary Use Case in Seagrass Monitoring | Key Challenges |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multispectral Satellites**<br>(*Sentinel-2, Landsat 8/9*) | 10–20 m | 5 days (S2 constellation) | Shallow to ~15–20 m (clear waters) | Basin-wide, national, and regional continuous monitoring; historical time-series (Landsat archive >30 yrs). | Mixed pixels in patchy meadows; attenuation with depth; cloud cover; atmospheric/air-water interface glint. |
| **High / Very-High Res Satellites**<br>(*PlanetScope, WorldView-2/3/4, Pléiades*) | 0.3–3 m | Daily to on-demand | Shallow to ~20–25 m | High-precision MPA mapping; identifying meadow blowouts, anchor damage, and coastal fragmentation. | High commercial cost; band-to-band registration jitter; lack of consistent radiometric calibration compared to Sentinel. |
| **Drones / UAVs**<br>(*RGB, Multispectral, Hyperspectral*) | Centimetric (1–5 cm) | On-demand | 0 to ~10–15 m | Patch-scale dynamics, shoot density proxies, shallow fringe quantification, *banquettes* (beached necromass), validation baselines. | Limited spatial coverage per flight; sun glint, wave refraction, and surface ripples; strict flight regulations in coastal/protected areas. |
| **Acoustic Sensors**<br>(*Side-Scan Sonar - SSS, Multibeam - MBES*) | Decimetric to metric | Campaign-based | Full depth range (down to lower limit >35–40 m) | Mapping lower depth limits, turbid Atlantic/coastal waters where optics fail; bathymetry + acoustic backscatter for substrate hardness/roughness. | Low acquisition speed; vessel cost; complex acoustic propagation and angular response dependence; no direct spectral color information. |
| **Underwater Imaging**<br>(*Drop-cam, Towed-cam, Diver Photoquadrats, ROV/AUV*) | Millimetric to centimetric | Campaign-based | Localized (ground truth) | Taxonomic species discrimination, health status (epiphytes, necrosis, flowering), shoot density calibration, benthic ground truth. | Extremely sparse spatial sampling; severe color cast (blue/green absorption); turbidity, floating suspended matter (*marine snow*); high operational costs. |

---

## 3. Phase I: Classical Machine Learning & Pixel-by-Pixel Classification

### 3.1 The Classical Workflow
Traditional pipelines treat each pixel as an independent multi-dimensional vector:
1. **Pre-processing:** Atmospheric correction (e.g., Sen2Cor, ACOLITE, Polymer) to obtain Bottom-of-Atmosphere (BOA) surface reflectance.
2. **Water Column Correction (WCC):** Crucial step to decouple seafloor reflectance from water attenuation:
   - *Lyzenga Depth-Invariant Indices (DII)* (Lyzenga 1978, 1981): Linear transformation using band pairs over homogenous substrate at varying depths.
   - *Stumpf Band Ratio Model* (Stumpf et al., 2003): Relative bathymetry estimation to normalize spectral decay.
   - *Semi-Analytical Inversion Models* (Lee et al., 1999; Maritorena et al.): Radiative transfer equations modeling inherent optical properties (IOPs: absorption $a$, backscattering $b_b$).
3. **Pixel Classifiers:**
   - **Maximum Likelihood Classifier (MLC):** Parametric Gaussian assumption.
   - **Support Vector Machines (SVM):** Kernel tricks (RBF) for non-linear decision boundaries; robust in small sample regimes.
   - **Random Forest (RF):** Ensemble of decision trees; dominant standard due to feature importance metrics, robustness to noise, and native handling of multi-band data.

### 3.2 Fundamental Limitations of Pixel-by-Pixel Approaches
* **No Spatial Context:** An isolated pixel cannot distinguish whether it belongs to a continuous meadow, an isolated patch, a rocky reef with dark photophilic algae, or a deep shadow.
* **Salt-and-Pepper Artifacts:** High intra-class spectral variance generates noisy, speckle-dominated classification maps requiring ad-hoc morphological filtering (opening/closing, majority filters).
* **Mixed Pixel Problem:** At 10 m resolution (Sentinel-2), a single pixel contains a mixture of *Posidonia*, sand, dead matte, and micro-patches. Pixel classifiers force a hard label or produce poorly calibrated soft probabilities.
* **Sensitivity to Water-Surface Perturbations:** Sun glint, floating debris, and wave lenses degrade single-pixel spectra dramatically without contextual cues to regularize the prediction.

### 3.3 Key References (Classical Era)
* **Lyzenga, D. R. (1978, 1981).** *Passive remote sensing techniques for mapping water depth and bottom features.* Applied Optics. *(Foundational physics of optical depth-invariant indices)*.
* **Stumpf, R. P., Holderied, K., & Sinclair, M. (2003).** *Determination of water depth with high-resolution satellite imagery over variable bottom types.* Limnology and Oceanography. *(Standard ratio-based bathymetry and benthic correction)*.
* **Traganos, D., & Reinartz, P. (2018).** *Mapping Mediterranean seagrasses with Sentinel-2 imagery: Paving the way for strict conservation and climate change mitigation.* Frontiers in Plant Science, 9, 1398. *(Landmark paper demonstrating cloud-based Google Earth Engine (GEE), Random Forest, and SVM for regional Posidonia mapping with Sentinel-2)*.
* **Poursanidis, D., Topouzelis, K., & Chrysoulakis, N. (2018).** *Mapping coastal marine habitats and delineating the deep limits of the Neptune grass meadows using Sentinel-2.* Remote Sensing, 10(6), 863. *(Demonstrated Random Forest on S2 with bathymetric constraints in the Mediterranean)*.
* **Fauzan, M. A., et al. (2021).** *Comparison of machine learning algorithms for seagrass mapping using PlanetScope imagery.* International Journal of Remote Sensing.

---

## 4. Phase II: Deep Learning & the Exploitation of Spatial Context

### 4.1 The Paradigm Shift: From Spectral Vectors to Spatial-Spectral Features
Instead of classifying $x \in \mathbb{R}^B$ (where $B$ is the number of spectral bands), Deep Learning processes image tensors $X \in \mathbb{R}^{H \times W \times B}$.
* **Receptive Fields & Hierarchical Features:** Convolutional filters extract low-level edges, intermediate meadow patch textures, and high-level contextual relationships (e.g., meadows typically grow adjacent to sand plains or rocky bathymetric slopes, not in deep open trenches).
* **Spatial Regularization:** The network naturally learns spatial continuity, eliminating the salt-and-pepper noise without arbitrary post-processing filters.

### 4.2 Key Architectures Applied to Benthic & Seagrass Mapping

#### A. Semantic Segmentation (Wall-to-Wall Dense Prediction)
* **U-Net & Variants (Attention U-Net, ResU-Net):**
  - Encoder-decoder with skip connections preserving fine-grained spatial boundaries (critical for fragmented meadow borders and anchor scars).
  - Widely used for both satellite tile segmentation and drone orthomosaic classification.
* **DeepLabv3+ (Atrous Spatial Pyramid Pooling - ASPP):**
  - Dilated convolutions capture multi-scale context without losing spatial resolution.
  - Excellent for capturing both vast continuous meadows (*macro-structure*) and isolated small shoots (*micro-structure*).
* **Feature Pyramid Networks (FPN) & HRNet:**
  - Maintain high-resolution representations throughout the network; ideal for resolving thin coastal fringes of *Cymodocea* or *Zostera*.
* **Vision Transformers for Segmentation (SegFormer, Swin-UNet):**
  - Replace convolutions with self-attention mechanisms; captures global spatial dependencies across large satellite scenes (e.g., basin-scale coastal geomorphology).

#### B. High-Resolution UAV Photogrammetry & Deep Learning
* Integration with Structure-from-Motion (SfM) point clouds:
  - UAVs yield Digital Surface Models (DSM) + multispectral orthomosaics.
  - Deep CNNs trained on centimetric orthomosaics predict **shoot density**, **canopy height**, and quantify **patch fragmentation** (perimeter-to-area ratios).
  - Monitoring anchor scarring in high-traffic anchorages (e.g., Ses Salines in Formentera, Cabrera Archipelago).

#### C. Underwater Computer Vision (ROV, AUV, Drop-Cameras)
* **CoralNet Evolution:** Transitioned from manual point annotation to automated patch classification using ResNet backbones.
* **Substrate & Health Semantic Segmentation:** Segmenting living *Posidonia oceanica* vs. *dead matte* (decayed rhizome matrix) vs. sand vs. invasive macroalgae (*Rugulopteryx okamurae*, *Caulerpa cylindracea*).
* **Color Restoration Pre-processing:** Integrating Sea-thru (Akkaynak & Treibitz) or CycleGANs/Water-GANs to reverse underwater light attenuation (red spectrum loss) before feeding images to classification backbones.

#### D. Acoustic Backscatter & Multibeam Bathymetry Deep Learning
* Using 2D CNNs on Side-Scan Sonar (SSS) acoustic mosaics and Multibeam Echosounder (MBES) backscatter intensity maps.
* Exploiting Angular Response Analysis (ARA) alongside deep spatial textures to detect seagrass meadows at depths (>25–40 m) where optical satellites reach the photic limit.

### 4.3 Key References (Deep Learning Era)
* **Gómez-Donoso, J., et al. (2020/2021).** *Automatic underwater classification of Posidonia oceanica using deep convolutional neural networks.* (Alicante/Spanish research initiatives on benthic imaging).
* **Montero-Hidalgo, M., et al. (2023).** *Fine-scale mapping of seagrass meadows using unmanned aerial vehicles (UAVs) and deep learning.* Estuarine, Coastal and Shelf Science.
* **Traganos, D., et al. (2022).** *Sub-meter classification of Posidonia oceanica using WorldView-3 and deep semantic segmentation.* Marine Pollution Bulletin.
* **Ventura, D., et al. (2022).** *Mapping and monitoring of Posidonia oceanica meadows through UAV photogrammetry and deep learning semantic segmentation.* Remote Sensing.
* **Akkaynak, D., & Treibitz, T. (2019).** *Sea-thru: A method for removing water from underwater images.* IEEE/CVF CVPR. *(Essential reference for restoring true benthic reflectance in imagery)*.
* **Conti, L. A., et al. (2021).** *Deep learning semantic segmentation of benthic habitats using acoustic and underwater optical data.* Marine Environmental Research.

---

## 5. Phase III: Foundation Models & Self-Supervised Learning (SSL)

### 5.1 The Benthic Ground-Truth Bottleneck
Why supervised deep learning reaches an operational roadblock in marine environments:
* **Scarcity & Cost of In Situ Data:** Terrestrial AI leverages millions of annotated images (ImageNet, OpenEarthMap). Marine benthic habitats require scientific scuba divers, research vessels, calibrated underwater cameras, or multibeam sonar—costing thousands of euros per operational day.
* **Sparse & Heterogeneous Labels:** In situ data is often collected as GPS point coordinates, single transect lines, or diver quadrats, while satellite and drone mapping requires dense, 2D continuous masks.
* **Label Noise & Taxonomic Ambiguity:** Differentiating dead matte covered by brown algae from living sparse *Posidonia* is subjective even among expert divers.

### 5.2 The Emerging Solutions

```
+---------------------------------------------------------------------------------------------------+
|                                 UNSUPERVISED PRE-TRAINING (PetaBytes)                             |
|  Global Sentinel-2 Water Scenes  |  Global ROV/AUV Video Archives  |  Global Sonar/Bathymetry     |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                  Self-Supervised Learning (SSL)
                      (Masked Autoencoding / Contrastive Vision-Language Learning)
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                     FOUNDATION MODELS / BACKBONES                                 |
|  * Earth Observation:  Prithvi (NASA/IBM), Clay Foundation, SatMAE, Hydro, OceanMAE               |
|  * Vision-Language:    BioCLIP (TreeOfLife), AquaticCLIP                                          |
|  * Promptable Vision:  Segment Anything Model (SAM), MarineSAM                                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                      Downstream Task Adaptation
                             (Linear Probing, LoRA / PEFT, Minimal Annotations)
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                                  OPERATIONAL SEAGRASS MONITORING                                  |
|     * Posidonia vs. Sand vs. Matte    * Shoot Density Estimation    * Basin-Scale Edge Detection  |
+---------------------------------------------------------------------------------------------------+
```

#### A. Self-Supervised Learning (SSL) in Remote Sensing
* **Masked Autoencoders (MAE / SatMAE):**
  - Large portions (75–80%) of multi-spectral or spatial image patches are masked. The model learns to reconstruct the missing spectral and structural information.
  - Learns rich spatial-spectral representations from millions of *unlabeled* Sentinel-2/UAV coastal scenes without a single manual label.
* **Contrastive Pre-training (MoCo, SimCLR):**
  - Enforces invariance to illumination, minor turbidity fluctuations, and surface waves by pulling augmented views of the same marine patch together in latent space.

#### B. Earth Observation Foundation Models
* **Prithvi (NASA / IBM / Hugging Face, 2023–2024):**
  - 100M parameter temporal vision transformer pre-trained on Harmonized Landsat-Sentinel (HLS) data. Fine-tunable with small amounts of seagrass ground truth.
* **Clay Foundation Model (Clay, 2024):**
  - Multi-sensor self-supervised foundation model for Earth observation with native support for spatial, temporal, and multi-band metadata.
* **Hydro & OceanMAE (2024–2025):**
  - Specifically designed foundation models pre-trained exclusively on global water bodies, marine coastal margins, and incorporating bathymetric and radiative transfer priors to solve the terrestrial bias of standard EO foundation models.

#### C. Biological & Underwater Vision-Language Models
* **BioCLIP (Imageomics Institute / CVPR 2024):**
  - Pre-trained on *TreeOfLife-10M*, incorporating hierarchical taxonomy across 450,000+ species.
  - Zero-shot classification and highly informative feature extraction for underwater benthic species (*Posidonia oceanica*, *Cymodocea nodosa*, *Caulerpa* spp.).
* **Segment Anything Model (SAM) & MarineSAM:**
  - Zero-shot / few-shot promptable segmentation (points, bounding boxes, text).
  - Dramatically accelerates expert annotation: a single point click on an ambiguous meadow generates precise pixel-level boundaries.
  - MarineSAM fine-tunes SAM with underwater marine prompts to combat low contrast and backscatter noise.

### 5.3 Key References (Foundation Models & SSL Era)
* **Stevens, S., et al. (CVPR 2024).** *BioCLIP: A Vision Foundation Model for the Tree of Life.*
* **Jakubik, J., et al. (NASA/IBM, 2023).** *Foundation Models for Generalist Geospatial Artificial Intelligence (Prithvi).* arXiv:2310.18660.
* **Cong, Y., et al. (2022/2023).** *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS.
* **Kirillov, A., et al. (Meta AI, 2023).** *Segment Anything.* IEEE/CVF ICCV.
* **Yan, X., et al. (2024).** *MarineSAM: Segment Anything in Marine Environments.*
* **Wang, Z., et al. (2024).** *OceanMAE / Hydro: Self-Supervised Masked Image Modeling for Water and Oceanographic Remote Sensing.*

---

## 6. Spanish Ecosystem, National Initiatives, & Local Focus

Since the congress is hosted in Spain, grounding the theoretical evolution in Spanish projects, ecosystems, and regulatory frameworks will resonate deeply with the audience.

### 6.1 Focal Seagrass Species in Spanish Waters
1. ***Posidonia oceanica* (Mediterranean):**
   - Endemic, slow-growing (1–3 cm/year), long-lived (centuries to millennia).
   - Massive carbon storage (*matte* up to several meters thick, millenary blue carbon deposits).
   - Priority habitat under the EU Habitats Directive (Code 1120*).
   - Primary hotspots: Balearic Islands (Ibiza and Formentera UNESCO World Heritage site), Cabo de Gata-Níjar (Andalucía), Murcia coast, Alicante and Columbretes Islands (Comunidad Valenciana), Cap de Creus / Medes (Catalunya).
2. ***Cymodocea nodosa* (*Sebadal* / *Gramaje*):**
   - Warmer Mediterranean and Atlantic waters; shallow bays, coastal lagoons (e.g., historical dynamics in Mar Menor), and Canary Islands (*Sebadales*).
3. ***Zostera marina* & *Zostera noltei* (*Nanostera noltii*):**
   - Dominant in Atlantic Spain: Galician Rías Baixas and Rías Altas, Cantabrian estuaries (e.g., Santander, Villaviciosa in Asturias), Gulf of Cádiz, and Doñana marshes.

### 6.2 Key Spanish Projects, Frameworks, and Working Groups
* **CartoGrass (Convenio IEO-CSIC & Universidades / MITECO):**
  - The nationwide framework aiming to update, harmonize, and standardize seagrass mapping and validation criteria across Spain.
  - Co-author Fernando García-González (IEO-CSIC) is a central researcher leading the remote sensing and validation criteria lines.
* ***Atlas de las Praderas de Fanerógamas Marinas de España* (Ruiz et al., 2015, IEO / MITECO):**
  - The benchmark national compilation of seagrass meadows in Spain. Part 1 of the talk highlights how AI and modern multi-sensor processing transform this static baseline into continuous, dynamic monitoring.
* **LIFE IP INTEMARES (2017–2024+):**
  - Coordinated by Fundación Biodiversidad (MITECO) with IEO-CSIC as a core partner. The largest marine conservation initiative in Europe, integrating satellite, acoustic, and in situ data to manage the Natura 2000 marine network.
* **LIFE Blue Natura Andalucía (LIFE14 CCM/ES/000957):**
  - Pioneering project quantifying blue carbon stocks in *Posidonia oceanica* meadows and saltmarshes in Andalucía, directly driving demand for automated extent/density mapping.
* **LIFE ECOREST (Catalonia) & Posidonia GOIB (Balearic Islands):**
  - Regional operational applications: The Balearic Government’s *Posidonia Maps* and surveillance fleets against illegal anchoring; automated Sentinel-2 and aerial monitoring systems.
* **Key Spanish Research Centers in Marine AI & Remote Sensing:**
  - **IEO-CSIC (Instituto Español de Oceanografía):** Seagrass Ecology Group (GEAM), Oceanographic Centers of Murcia, Gijón, Santander, Palma, etc.
  - **AIC (Artificial Intelligence Center, Universidad de Oviedo):** Machine learning, distribution shifts, transfer learning.
  - **IMEDEA (CSIC-UIB, Mallorca):** Global leaders in seagrass ecology and blue carbon (Carlos Duarte, Núria Marbà).
  - **CEAB-CSIC (Blanes):** Benthic ecology, biogeochemistry, and long-term acoustic/benthic monitoring.
  - **IPL (Image and Signal Processing Group, Universitat de València - Gustau Camps-Valls):** International references in AI, physics-aware machine learning, and Earth observation.
  - **ViCOROB (University of Girona):** World leaders in underwater robotics (AUVs: Girona 500, Iver) and optical/acoustic seafloor mapping.

---

## 7. Slide-by-Slide Outline for Part 1 (Ready for Slide Deck)

### Slide 1: Title & Overview
* **Title:** Artificial Intelligence Applied to the Mapping and Monitoring of Seagrass Meadows.
* **Speakers:** Pablo González (AIC, UniOvi) & Fernando García-González (IEO-CSIC).
* **Takeaway:** Introducing the two-part structure: (1) The AI evolutionary journey (From Pixels to Foundation Models); (2) The operational frontier: Handling Dataset Shift.

### Slide 2: The Ecological & Regulatory Mandate
* Ecological services: Blue carbon sequestration, coastal buffer against wave energy, biodiversity hot spots.
* Legal frameworks: EU Habitats Directive (1120*), Marine Strategy Framework Directive (MSFD), Law 42/2007 in Spain, Balearic Posidonia Decree.
* The limitation: In situ sampling is high-precision but spatially sparse and expensive. We need scalable observation.

### Slide 3: Multi-Sensor Remote Sensing Ecosystem
* Visual graphic comparing the 4 pillars:
  - **Satellites (Sentinel-2, PlanetScope):** Macroscopic, continuous, multi-temporal.
  - **Drones (UAVs):** Centimetric, patch-scale, canopy structure, rapid deployment.
  - **Acoustics (MBES, SSS):** Penetrating beyond the photic zone, deep limit (>35 m), acoustic backscatter for substrate hardness.
  - **Underwater Robotics & Vision (ROV/AUV/Drop-cam):** Species-level ground truthing, health metrics (epiphytes, necrosis).

### Slide 4: Era 1 — Classical Pixel-by-Pixel Classification
* The workflow: Top-of-Atmosphere $\rightarrow$ Atmospheric Correction $\rightarrow$ Water Column Correction (Lyzenga DII, Stumpf ratio) $\rightarrow$ Classifier (MLC, SVM, Random Forest).
* Breakthrough of the 2010s: Cloud computing on Google Earth Engine (GEE) + Sentinel-2 enabled regional-scale maps (Traganos et al., Poursanidis et al.).

### Slide 5: The Ceiling of Pixel-Based Methods
* Why pixels fail:
  - High intra-class spectral variability (sun glint, turbidity spikes, leaf angle).
  - No spatial context: cannot differentiate an isolated clump of macroalgae on a rock from a degraded seagrass meadow.
  - The "salt-and-pepper" noise dilemma and failure at transitional boundaries.

### Slide 6: Era 2 — Deep Learning & Spatial-Spectral Context
* The core idea: Moving from spectral vectors to 2D image patches and semantic segmentation.
* Architectures: U-Net (preserving fine meadow edges and anchor scars) and DeepLabv3+ (multi-scale context via atrous convolutions).
* Eliminating speckle noise naturally: The model learns spatial continuity and benthic geomorphology.

### Slide 7: Deep Learning Across Scales: Drones & Underwater Vision
* **Drones:** UAV multispectral orthomosaics + CNNs for fine-scale patchiness, fragmentation index, and beach-cast *banquettes*.
* **Underwater AI:** Automated quadrat analysis (CoralNet, ResNet/VGG), differentiating live *Posidonia* from *dead matte* and detecting invasive threats (*Rugulopteryx okamurae*).

### Slide 8: Era 3 — The Ground-Truth Bottleneck & Self-Supervised Learning (SSL)
* The fundamental dilemma: Deep learning is data-hungry, but marine pixel annotations cost thousands of euros in ship/diver time.
* The solution: **Self-Supervised Learning (SSL)**.
* Masked Autoencoders (MAE): Teaching models to reconstruct masked marine scenes across millions of unannotated Sentinel-2 tiles.

### Slide 9: Foundation Models for Benthic & Marine Systems
* **Earth Observation:** Prithvi (NASA/IBM), Clay, OceanMAE/Hydro.
* **Vision-Language & Bio-intelligence:** BioCLIP (zero-shot taxonomic recognition of marine flora).
* **Interactive Segmentation:** Segment Anything Model (SAM) and MarineSAM for semi-automated annotator workflows.
* The new paradigm: Foundation model pre-training $\rightarrow$ parameter-efficient fine-tuning (LoRA / linear probing) with minimal in situ labels.

### Slide 10: The Spanish Scene — From the *Atlas* to CartoGrass
* Historical baseline: *Atlas de las Praderas de Fanerógamas Marinas de España* (Ruiz et al., 2015).
* Ongoing revolution: The **CartoGrass** initiative and **LIFE IP INTEMARES**.
* Bridging national mapping with modern AI pipelines developed at IEO-CSIC and Spanish academic centers.

### Slide 11: The Natural Transition to Part 2 (Dataset Shift)
* "Now we have deep spatial models and powerful foundation models... do they work when we deploy them in the real ocean?"
* Teaser for Part 2:
  - Seasonality and leaf shedding (summer vs. winter canopy).
  - Water turbidity spikes after storms.
  - Geographical transfer: Training in the crystalline waters of Formentera and inferring in the turbid Atlantic Rías of Galicia or the Mar Menor.
  - Sensor mismatch: Transferring from Sentinel-2 to PlanetScope or UAV.
* *Pass the floor to Part 2: Formally addressing Dataset Shift in Seagrass Monitoring.*

---

## 8. Curated Bibliography & Reading List for Slide Preparation

### A. Foundational Remote Sensing & Water Column Correction
1. **Lyzenga, D. R. (1981).** *Reflectance of a flat shallow ocean bottom: an formula for calculating bottom reflectance from remote sensing data.* International Journal of Remote Sensing, 2(1), 71-77.
2. **Stumpf, R. P., Holderied, K., & Sinclair, M. (2003).** *Determination of water depth with high-resolution satellite imagery over variable bottom types.* Limnology and Oceanography, 48(1part2), 547-555.
3. **Lee, Z., et al. (1999).** *Hyperspectral remote sensing for shallow waters: 2. Deriving bottom depths and water properties by optimization.* Applied Optics, 38(18), 3831-3843.

### B. Classical Machine Learning on Satellites
4. **Traganos, D., & Reinartz, P. (2018).** *Mapping Mediterranean seagrasses with Sentinel-2 imagery: Paving the way for strict conservation and climate change mitigation.* Frontiers in Plant Science, 9, 1398.
5. **Poursanidis, D., Topouzelis, K., & Chrysoulakis, N. (2018).** *Mapping coastal marine habitats and delineating the deep limits of the Neptune grass meadows using Sentinel-2.* Remote Sensing, 10(6), 863.
6. **Kovacs, E., et al. (2018).** *Seagrass mapping using high-resolution satellite imagery and object-based classification.* Marine Pollution Bulletin, 134, 155-164.

### C. Deep Learning, UAVs & High-Resolution Benthic Mapping
7. **Montero-Hidalgo, M., et al. (2023).** *Fine-scale mapping of seagrass meadows using unmanned aerial vehicles (UAVs) and deep learning.* Estuarine, Coastal and Shelf Science, 281, 108200.
8. **Ventura, D., et al. (2022).** *Mapping and monitoring of Posidonia oceanica meadows through UAV photogrammetry and deep learning semantic segmentation.* Remote Sensing, 15(1), 173.
9. **Traganos, D., et al. (2022).** *Sub-meter classification of Posidonia oceanica using WorldView-3 and deep semantic segmentation.* Marine Pollution Bulletin, 185, 114298.
10. **Beijbom, O., et al. (2015).** *Towards automated annotation of benthic survey images: CoralNet.* Frontiers in Marine Science, 2, 108.
11. **Akkaynak, D., & Treibitz, T. (2019).** *Sea-thru: A method for removing water from underwater images.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1682-1691.

### D. Foundation Models & Self-Supervised Learning
12. **Stevens, S., Wu, J., Thompson, M. J., Campolongo, E. G., Song, C. H., et al. (2024).** *BioCLIP: A Vision Foundation Model for the Tree of Life.* IEEE/CVF CVPR 2024.
13. **Jakubik, J., et al. (2023).** *Foundation Models for Generalist Geospatial Artificial Intelligence (Prithvi).* arXiv:2310.18660.
14. **Kirillov, A., et al. (2023).** *Segment Anything.* IEEE/CVF International Conference on Computer Vision (ICCV), 4015-4026.
15. **Yan, X., et al. (2024).** *MarineSAM: Segment Anything in Marine Environments.* arXiv:2404.09988.
16. **Cong, Y., et al. (2022).** *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* Advances in Neural Information Processing Systems (NeurIPS), 35, 197-211.

### E. Spanish Context, Atlases & Management
17. **Ruiz, J. M., Guillén, J. E., Ramos-Segura, A., & Otero, M. M. (Eds.). (2015).** *Atlas de las praderas de fanerógamas marinas de España.* IEO/IEL/UICN, Murcia-Palma de Mallorca, 681 pp.
18. **García-González, F., et al. (CartoGrass & IEO-CSIC technical reports).** *Cartographic validation criteria and methods applied to seagrass meadows monitoring.*
19. **Marbà, N., & Duarte, C. M. (2010).** *Mediterranean seagrass (Posidonia oceanica) loss between 1842 and 2009.* Biological Conservation, 143(6), 1459-1468.
20. **Telesca, L., et al. (2015).** *Seagrass meadows (Posidonia oceanica) distribution and trajectories of change in the Mediterranean Sea.* Advances in Marine Biology, 72, 285-300.

