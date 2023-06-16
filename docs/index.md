---
title: Breast Cancer Detection (Group 5)
layout: default
---
# Introduction
Breast cancer is one of the most common cancers worldwide, with 284,200 breast cancer cases and 44,130 deaths in 2021 in the US alone [1]. Early detection and accurate classification of breast tissues as either cancerous or benign are crucial for timely and effective treatment. While these classifications are currently performed by trained medical professionals from histopathological data, over the years, researchers have made significant progress in developing machine-learning techniques to diagnose breast cancer. These techniques leverage features from medical imaging data, such as mammograms and ultrasound images from breast cancer patients [2].  Generally, the classification distinguishes breast tumors into two types, benign and malignant. Benign is a noninvasive (non-cancerous) while malignant is an invasive (cancerous) type of tumor. But, both tumors have further subtypes that must be diagnosed individually because each may lead to different prognoses and treatment plans. The current problem lies in the variability and complexity of breast tissue characteristics, making accurate classification a non-trivial task. These subtypes are often detected by cellular measurements, such as the cell's radius, cell texture, the shape of nuclei, etc., and proper diagnosis requires accurate identification of each subcategory of breast cancer [3].

# Problem Statement
Despite the advancements in breast cancer classification, there still remains a challenge to achieving high accuracy and reliability. The motivation behind this project is to develop an ML model that not only accurately classifies breast tissues as cancerous or benign but also summarizes the most important features driving the decisions (a more game theory approach of calculating Shapely values for the cell features) [4]. 

# Dataset

We will use the publicly available Wisconsin Breast Cancer Diagnostic dataset [5], with samples labeled “benign” (357 samples) or “malignant” (212 samples), The features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass and describe the characteristics of the cell nuclei in the image. The 10 real-valued features computed for each cell nucleus are:
a) radius (mean of distances from the center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)


# Methods

# Potential Results and Discussion

# References

> Siegel, R. L., Miller, K. D., Fuchs, H. E., & Jemal, A. (2021). Cancer statistics, 2021. CA: a cancer journal for clinicians, 71(1), 7-33

> Aksebzeci BH, Kayaalti Ö (2017) Computer-aided classification of breast cancer histopathological images. Paper presented at the 2017 Medical Technologies National Congress (TIPTEKNO)

> Murtaza, G., Shuib, L., Abdul Wahab, A.W. et al. Deep learning-based breast cancer classification through medical imaging modalities: state of the art and research challenges

> Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Advances in Neural Information Processing Systems. 2017

> Wisconsin Diagnostic Breast Cancer (WDBC) Dataset and Wisconsin Prognostic Breast Cancer (WPBC) Dataset.
http://ftp.ics.uci.edu/pub/machine-learning-databases/breast-cancer-wisconsin/
