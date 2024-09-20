# **Micro-Particle Signal Segmentation in Self-Mixing Interferometry Signals**

![License](https://img.shields.io/badge/license-MIT-green.svg)  

This project focuses on segmenting induced modulation signals from an SMI sensor as microparticles pass through the laser sensing volume. It establishes the initial signal processing for acquisition and sets the stage for further processing and machine learning algorithms for single-particle classification.

This repository contains the code, datasets, and instructions for reproducing the results presented in the paper: <br>
**Tittle:** [Adaptive Single Micro-Particle Detection and Segmentation in Self-Mixing Interferometry Signal] <br>
**Authors:** [Sebastián Sierra-Alarcón, Julien Perchoux, Francis Jayat, Clement Tronche, Santiago S. Pérez and Adam Quotb]<br>
 In progress to be published: [Journal/Conference Name], [Year]

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## **Introduction**

This repository contains the main codes involved in the signal processing techniques for segmenting the signals for the SMi sensor, thinking it can be used as a basis for further research in the use of this technique for particle and cell classification. Additionally, if you´re interesting in the full dataset, you can contact us; we are oppening to work with more laboratories and research teams.


## **Project Structure**



## **Installation**

### Prerequisites
- [Python >= 3.11]

### Steps
1. **Clone the repository:**
    ```bash
    git clone https://github.com/SSierraAl/SMI-ParticleSegmentation.git
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
    
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
## **Usage**

- **Data Preprocessing:**  
    Run the data preprocessing script to prepare all the datasets.
    ```bash
    python main.py --input Particles_Data/Raw_Xum --output Particles_Data/DB_Xum
    ```

- **Visualization:**  
    If you want to obserb the general steps without go deeper you can check the following notebook.
    Visualization.ipynb
    

## **Contributing**

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Contact**

For any questions or issues, please contact:

- **[Sebastian Sierra ]** - [sebsieal@hotmail.com]
- **GitHub Issues:** Open an issue on the repository

