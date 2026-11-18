==========
References
==========

Each simulator and metric in Pasteur follows from published work on
missing data, laboratory measurement, EHR data quality, dataset shift and
software testing. This page says which, so a reader can check the reasoning
and judge where it stops applying.

A citation here explains why a perturbation or metric is shaped the way it
is. It does not mean the cited authors endorse Pasteur, or that a Pasteur
number reproduces a result from their paper.

Simulators
----------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Simulator
     - Why it is built this way
     - Sources
   * - Blackout
     - Nulling a uniform random share of rows is missingness completely at
       random, the simplest mechanism in Rubin's taxonomy. Real EHR
       missingness is rarely that: whether a test was ordered carries
       information about the patient, often more than its value. Blackout
       is therefore a floor, and companions keep a row internally
       consistent when its measured flag is itself a model input. How
       missing values should be handled also differs between prediction and
       explanation.
     - [Rubin1976]_, [Agniel2018]_, [Sperrin2020]_
   * - Jitter
     - A laboratory result varies between draws through analytical
       imprecision and within-person biological variation, with no change in
       the patient's state. A model's answer should not depend on which draw
       it saw.
     - [Fraser1989]_
   * - Flipper
     - Straight-line paths between two real patients with different labels
       probe the model between observed points, as mixup does in training,
       and locate the smallest change that alters the decision, as
       counterfactual explanations do.
     - [Zhang2018]_, [Wachter2018]_
   * - Sweep
     - Moving one feature while the rest of the row stays fixed traces an
       individual conditional expectation curve for each patient.
     - [Goldstein2015]_
   * - Scramble
     - The encodings are the reasons laboratory results were found not to
       standardise across a multicentre data warehouse: inequality signs
       attached to numbers, text in numeric fields, and typographic
       variants. Conformance, completeness and plausibility are the
       standard categories for such defects, and ``parse_loss`` reports the
       first of them turning into the second.
     - [Hauser2018]_, [Kahn2016]_, [Weiskopf2013]_
   * - Units
     - Sites report the same analyte in conventional or SI units. Most
       conversions are a factor; HbA1c is affine, by the IFCC–NGSP master
       equation ``NGSP = 0.09148 × IFCC + 2.152``, which the
       ``hba1c_ngsp_to_ifcc`` preset inverts.
     - [Hoelzel2004]_
   * - Site
     - Clinical models lose performance between hospitals for reasons tied
       to each site's processes and data handling rather than to biology. A
       site composes several such differences, and stability under a
       declared shift can be estimated from the data at hand.
     - [Zech2018]_, [Wong2021]_, [Finlayson2021]_, [Subbaswamy2021]_
   * - Site draws
     - Domain randomization: evaluate over many randomized variants of the
       simulator so that the real environment looks like one more variant.
     - [Tobin2017]_

Metrics
-------

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Metric
     - Why it is defined this way
     - Sources
   * - ``generalizability`` as the worst site
     - Average performance hides the group a model fails on, so robustness
       under shift is reported for the worst group or held-out domain. In
       clinical prediction, the site is the natural unit to hold out.
     - [Sagawa2020]_, [Koh2021]_, [Steyerberg2016]_
   * - ``decision_change_rate``
     - A change that keeps the patient's meaning should keep the answer.
       That is a metamorphic relation, tested without knowing the correct
       output; behavioural testing of NLP models calls it an invariance
       test, and image-corruption benchmarks report the same quantity as
       flip probability.
     - [Chen1998]_, [Segura2016]_, [Ribeiro2020]_, [Hendrycks2019]_
   * - ``silent_change_rate``, ``--abstain-class``
     - Declining to answer is a legitimate output when the input cannot be
       trusted. The reject option trades coverage for error, and what
       matters is the error on the cases the model chose to answer.
     - [Chow1970]_, [Geifman2017]_

Limits
------

Simulated robustness is evidence under declared conditions. It does not show
that a model generalises to a hospital it has not seen, and the clinical
literature is clear that this has to be established locally and reported in
full [Futoma2020]_ [Collins2024]_.

Bibliography
------------

.. [Agniel2018] Agniel D, Kohane IS, Weber GM. Biases in electronic health
   record data due to processes within the healthcare system: retrospective
   observational study. *BMJ*. 2018;361:k1479.
   https://www.bmj.com/content/361/bmj.k1479

.. [Chen1998] Chen TY, Cheung SC, Yiu SM. Metamorphic testing: a new approach
   for generating next test cases. Technical Report HKUST-CS98-01, Department
   of Computer Science, The Hong Kong University of Science and Technology;
   1998. https://arxiv.org/abs/2002.12543

.. [Chow1970] Chow CK. On optimum recognition error and reject tradeoff.
   *IEEE Transactions on Information Theory*. 1970;16(1):41-46.
   https://doi.org/10.1109/tit.1970.1054406

.. [Collins2024] Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement:
   updated guidance for reporting clinical prediction models that use
   regression or machine learning methods. *BMJ*. 2024;385:e078378.
   https://doi.org/10.1136/bmj-2023-078378

.. [Finlayson2021] Finlayson SG, Subbaswamy A, Singh K, et al. The clinician
   and dataset shift in artificial intelligence. *New England Journal of
   Medicine*. 2021;385(3):283-286. https://doi.org/10.1056/NEJMc2104626

.. [Fraser1989] Fraser CG, Harris EK. Generation and application of data on
   biological variation in clinical chemistry. *Critical Reviews in Clinical
   Laboratory Sciences*. 1989;27(5):409-437.
   https://doi.org/10.3109/10408368909106595

.. [Futoma2020] Futoma J, Simons M, Panch T, Doshi-Velez F, Celi LA. The myth
   of generalisability in clinical research and machine learning in health
   care. *The Lancet Digital Health*. 2020;2(9):e489-e492.
   https://doi.org/10.1016/S2589-7500(20)30186-2

.. [Geifman2017] Geifman Y, El-Yaniv R. Selective classification for deep
   neural networks. In: *Advances in Neural Information Processing Systems
   30*; 2017.
   https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

.. [Goldstein2015] Goldstein A, Kapelner A, Bleich J, Pitkin E. Peeking
   inside the black box: visualizing statistical learning with plots of
   individual conditional expectation. *Journal of Computational and
   Graphical Statistics*. 2015;24(1):44-65.
   https://doi.org/10.1080/10618600.2014.907095

.. [Hauser2018] Hauser RG, Quine DB, Ryder A. LabRS: a Rosetta stone for
   retrospective standardization of clinical laboratory test results.
   *Journal of the American Medical Informatics Association*.
   2018;25(2):121-126. https://doi.org/10.1093/jamia/ocx046

.. [Hendrycks2019] Hendrycks D, Dietterich T. Benchmarking neural network
   robustness to common corruptions and perturbations. In: *International
   Conference on Learning Representations*; 2019.
   https://arxiv.org/abs/1903.12261

.. [Hoelzel2004] Hoelzel W, et al. IFCC reference
   system for measurement of hemoglobin A1c in human blood and the national
   standardization schemes in the United States, Japan, and Sweden: a
   method-comparison study. *Clinical Chemistry*. 2004;50:166-174.
   https://doi.org/10.1373/clinchem.2003.024802

.. [Kahn2016] Kahn MG, et al. A harmonized data
   quality assessment terminology and framework for the secondary use of
   electronic health record data. *eGEMs*. 2016;4(1):1244.
   https://doi.org/10.13063/2327-9214.1244

.. [Koh2021] Koh PW, Sagawa S, Marklund H, et al. WILDS: a benchmark of
   in-the-wild distribution shifts. In: *Proceedings of the 38th
   International Conference on Machine Learning*. PMLR 139:5637-5664; 2021.
   https://proceedings.mlr.press/v139/koh21a.html

.. [Ribeiro2020] Ribeiro MT, Wu T, Guestrin C, Singh S. Beyond accuracy:
   behavioral testing of NLP models with CheckList. In: *Proceedings of the
   58th Annual Meeting of the Association for Computational Linguistics*;
   2020:4902-4912. https://doi.org/10.18653/v1/2020.acl-main.442

.. [Rubin1976] Rubin DB. Inference and missing data. *Biometrika*.
   1976;63(3):581-592. https://doi.org/10.1093/biomet/63.3.581

.. [Sagawa2020] Sagawa S, Koh PW, Hashimoto TB, Liang P. Distributionally
   robust neural networks for group shifts: on the importance of
   regularization for worst-case generalization. In: *International
   Conference on Learning Representations*; 2020.
   https://arxiv.org/abs/1911.08731

.. [Segura2016] Segura S, Fraser G, Sanchez AB, Ruiz-Cortés A. A survey on
   metamorphic testing. *IEEE Transactions on Software Engineering*.
   2016;42(9):805-824. https://doi.org/10.1109/TSE.2016.2532875

.. [Sperrin2020] Sperrin M, Martin GP, Sisk R, Peek N. Missing data should be
   handled differently for prediction than for description or causal
   explanation. *Journal of Clinical Epidemiology*. 2020;125:183-187.
   https://doi.org/10.1016/j.jclinepi.2020.03.028

.. [Steyerberg2016] Steyerberg EW, Harrell FE Jr. Prediction models need
   appropriate internal, internal-external, and external validation.
   *Journal of Clinical Epidemiology*. 2016;69:245-247.
   https://doi.org/10.1016/j.jclinepi.2015.04.005

.. [Subbaswamy2021] Subbaswamy A, Adams R, Saria S. Evaluating model
   robustness and stability to dataset shift. In: *Proceedings of the 24th
   International Conference on Artificial Intelligence and Statistics*. PMLR
   130:2611-2619; 2021. https://proceedings.mlr.press/v130/subbaswamy21a.html

.. [Tobin2017] Tobin J, Fong R, Ray A, Schneider J, Zaremba W, Abbeel P.
   Domain randomization for transferring deep neural networks from simulation
   to the real world. In: *2017 IEEE/RSJ International Conference on
   Intelligent Robots and Systems (IROS)*; 2017:23-30.
   https://doi.org/10.1109/iros.2017.8202133

.. [Wachter2018] Wachter S, Mittelstadt B, Russell C. Counterfactual
   explanations without opening the black box: automated decisions and the
   GDPR. *Harvard Journal of Law & Technology*. 2018;31(2):841-887.
   https://arxiv.org/abs/1711.00399

.. [Weiskopf2013] Weiskopf NG, Weng C. Methods and dimensions of electronic
   health record data quality assessment: enabling reuse for clinical
   research. *Journal of the American Medical Informatics Association*.
   2013;20(1):144-151. https://doi.org/10.1136/amiajnl-2011-000681

.. [Wong2021] Wong A, Otles E, Donnelly JP, et al. External validation of a
   widely implemented proprietary sepsis prediction model in hospitalized
   patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070.
   https://doi.org/10.1001/jamainternmed.2021.2626

.. [Zech2018] Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK.
   Variable generalization performance of a deep learning model to detect
   pneumonia in chest radiographs: a cross-sectional study. *PLoS Medicine*.
   2018;15(11):e1002683. https://doi.org/10.1371/journal.pmed.1002683

.. [Zhang2018] Zhang H, Cisse M, Dauphin YN, Lopez-Paz D. mixup: beyond
   empirical risk minimization. In: *International Conference on Learning
   Representations*; 2018. https://arxiv.org/abs/1710.09412
