import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
from chromadb import EmbeddingFunction, Documents, Embeddings
from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm

def init_db_client(user, pswd, db=None):
    if db:
        conn = mysql.connector.connect(
            host="localhost",
            user=user,
            password=pswd,
            database=db
        )
        return conn, conn.cursor()
    else:
         conn = mysql.connector.connect(
              host="localhost",
              user=user,
              password=pswd,
         )
         return conn, conn.cursor()

def split_clinicaltrials_data(embedding_model, data):
        # data = (id, var1, var2, ..., varn)
        if embedding_model == "neuml/pubmedbert-base-embeddings":
            chunk_size = 512
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

        combined = "".join(data[1:]) # combine values into single string

        docs, ids = [], []
        id = data[0]
        texts = splitter.split_text(combined)
        for i, t in enumerate(texts):
            ids.append(id+"_"+str(i))
            docs.append(t)
        return ids, docs

def scrape_huntsman_publications():
    data_path = "D:\\projects\\biochat\\pubmed\\publications"

    # get lab names
    labs = ['ayer', 'beckerle', 'bernard', 'cairns-lab', 'camp', 'chandrasekharan',
            'cheshier', 'curtin', 'doherty', 'edgar', 'evason', 'gaffney', 'gertz',
            'graves', 'grossman', 'hashibe', 'holmen', 'hu-lieskovan', 'jensen-lab',
            'johnson', 'kb-jones-lab', 'kaphingst', 'kepka', 'kinsey', 'kirchhoff',
            'mcmahon', 'mendoza', 'mooney', 'neklason', 'onega', 'schiffman', 'snyder',
            'spike', 'stewart', 'suneja', 'tavtigian', 'ullman', 'vanbrocklin', 'varley',
            'young', 'zhang']

    unique_urls = {'anderson': 'https://www.joshandersenlab.com/publications',
                   'basham': 'https://www.bashamlab.com/publications',
                   'buckley': 'https://buckleylab.org/',
                   'allie-grossmann': 'https://medicine.utah.edu/pathology/research-labs/allie-grossmann',
                   'torres': 'https://www.judsontorreslab.org/publications', # TO Fix
                   'myers': 'http://www.myerslab.org/publications.html',
                   'tan': 'http://tanlab.org/papers.html',
                   'welm-a': 'https://pubmed.ncbi.nlm.nih.gov/?term=Welm-A&show_snippets=off&sort=date&sort_order=asc&size=100',
                   'welm-b': 'https://pubmed.ncbi.nlm.nih.gov/?term=Welm-B&show_snippets=off&sort=date&sort_order=asc&size=100',
                   'wu': 'https://uofuhealth.utah.edu/huntsman/labs/wu/publications',
                   'ulrich' : 'https://uofuhealth.utah.edu/huntsman/labs/ulrich/publications'}

    # scrape publications from standard page format
    for lab in tqdm(labs):
        url = "https://uofuhealth.utah.edu/huntsman/labs/{}/publications".format(lab)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', class_='coh-wysiwyg')

        texts = []
        title = False
        for tag in div.find_all('p'):
            if lab in {'onega', 'young', 'mcmahon', 'edgar', 'johnson'}: # plain text
                text = tag.text.strip()
                is_header = (text.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((text.split()[0] + text.split()[1]) in {'Toview', 'Labmembers'})

            elif lab == 'ayer': # handle duplicate titles
                content = tag.text.strip()
                is_header = (content.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((content.split()[0] + content.split()[1]) in {'Toview', 'Labmembers'})
                if not is_header:
                    text = content.split('\n')[1]

            elif tag.find('a'): # standard format
                text = tag.text.strip()
                is_header = (text.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((text.split()[0] + text.split()[1]) in {'Toview', 'Labmembers'})
            
            if not is_header:
                texts.append(text.replace("\n", ""))

        path = os.path.join(data_path, "{}.txt".format(lab.replace("-", "_")))
        with open(path, "w", encoding='utf-8') as file:
            for t in texts:
                file.write(str(t) + "\n")

    # scrape from individual sites
    for lab in tqdm(unique_urls.keys()):
        url = unique_urls[lab]
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if lab == "anderson":
            div = soup.find_all('div', class_="sqs-html-content")[1]
            texts = []
            group = []
            for p in div.find_all('p'):
                if p.text.strip():
                    group.append(p.text.strip())
                elif group: # if p is empty and group is non-empty
                    texts.append(". ".join(group))
                    group = []
            if group: # add last group
                texts.append(". ".join(group))

            path = os.path.join(data_path, "anderson.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")
        
        if lab == "basham":
            div = soup.find('div', id='Containerc24vq')
            texts = []
            for p in div.find_all('p', class_='font_7'):
                text = p.text.strip()
                texts.append(text)
            
            path = os.path.join(data_path, "basham.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")

        if lab == "buckley":
            div = soup.find_all('div', class_='col-sm-10 col-sm-offset-1')[1]
            texts = []
            for p in div.find_all('p')[:-1]:
                if p.text.strip() != "(Selected Publications Since 2010)":
                    text = p.text.strip()
                    texts.append(text.lstrip('-\t').strip())
            path = os.path.join(data_path, "buckley.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")
        
        if lab == "allie-grossman":
            div = soup.find_all('div', class_='coh-wysiwyg')[-1]
            texts = []
            for p in div.find_all('li'):
                text = p.text.strip()
                texts.append(text)
            path = os.path.join(data_path, "allie_grossman.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")

        if lab == "torres": # TO DO
            publications = [
                "Genetic silencing of AKT induces melanoma cell death via mTOR suppression. Parkman GL, Turapov T, Kircher DA, Burnett WJ, Stehn CM, O'Toole K, Culver KM, Chadwick AT, Elmer RC, Flaherty R, Stanley KA, Foth M, Lum DH, Judson-Torres RL, Friend JE, VanBrocklin MW, McMahon M, Holmen SL. Mol Cancer Ther. 2023 Nov 3. doi: 10.1158/1535-7163.MCT-23-0474. Epub ahead of print. PMID: 37931033.",
                "BRAF variant allele fraction-A predictor of response to targeted therapy? Mitchel S Stark, Teresa Harris, Douglas Grossman, Robert L Judson-Torres. J Eur Acad Dermatol Venereol. 2023 Oct;37(10):1945-1946. doi: 10.1111/jdv.19389.",
                "Clinical utilization of olaparib, a PARP inhibitor, in BRCA1-mutant metastatic acral melanoma. Ruizin Jiang, Xianbin Liang, Ye Tao, Ronghui Xia, Ming Lei, Bin Jiang, Robert L Judson-Torres, Yanjie Zhang, Weixhen Zhang, Hanlin Zeng. Genes Dis. 2022 Dec 24;10(5):1755-1758. doi: 10.1016/j.gendis.2022.11.014.",
                "Natural resistance to cancer: A window of hope. Mohammad Masoudi, Parisa Torabi, Robert L Judson-Torres, Reza Khodarahmi, Sharif Moradi. International Journal of Cancer. 2023 Oct 20. doi: 10.1002/ijc.34766.",
                "Perilesional Epigenomes Distinguish Melanocytic Nevus Subtypes. Michael T Scherzer, Dekker C Deacon, Robert L Judson-Torres. Journal of Investigative Dermatology. 2023 Sep;143(9):1631-1633. doi: 10.1016/j.jid.2023.03.1670. Epub 2023 Jul 12.",
                "Critical Considerations for Investigating MicroRNAs during Tumorigenesis: A Case Study in Conceptual and Contextual Nuances of miR-211-5p in Melanoma. Fatemeh Vand-Rajabpour, Meghan Savage, Rachel L Belote, Robert L Judson-Torres. Epigenomes. 2023 Apr 26;7(2):9. doi: 10.3390/epigenomes7020009.",
                "Fabrication and validation of an LED array microscope for multimodal, quantitative imaging. Moustafa TE, Polanco ER, Belote RL, Judson-Torres RL, Zangle TA. HardwareX. 2023 Jan 24;13:e00399. doi: 10.1016/j.ohx.2023.e00399. PMID: 36756350; PMCID: PMC9900438.",
                "Quantitative Phase Imaging: Recent Advances and Expanding Potential in Biomedicine. Thang L. Nguyen, Soora Pradeep, Robert L. Judson-Torres, Jason Reed, Michael A. Teitell and Thomas A Zangle. ACS Nano 2022 Aug 16:8:11516-11544, /doi.org/10.1021/acsnano.1c11507.",
                "BRAFV600E induces reversible mitotic arrest in human melanocytes via microRNA-mediated suppression of AURKB. Andrew S. McNeal, Rachel L. Belote, Hanlin Zeng, Kendra Barker, Rodrigo Torres, Meghan Curtin, A. Hunter Shain, Robert H. I. Andtbacka, Sheri L. Holmen, David H. Lum, Timothy H. McCalmont, Matthew W. VanBrocklin, Douglas Grossman, Maria L. Wei, Ursula E. Lang, Robert L. Judson-Torres. eLife. 2021, 10:e70385.",
                "Human Melanocyte Development and Melanoma Dedifferentiation at Single-Cell Resolution. Rachel L. Belote, Daniel Le, Ashley Maynard, Ursula E. Lang, Adriane Sinclair, Brian K Lohman, Vicente Planells-Palop, Laurence Baskin, Aaron D. Tward, Spyros Darmanis, Robert L. Judson-Torres. Nat Cell Biol. 2021 Sep;23(9):1035-1047.",
                "Loon: Using Exemplars to Visualize Large-Scale Microscopy Data. Devin Lange, Eddie Polanco, Robert L Judson-Torres, Thomas Zangle, Alexander Lex. EEE Transactions on Visualization and Computer Graphics, vol. 28, no. 1, pp. 248-258, Jan. 2022, doi: 10.1109/TVCG.2021.3114766.",
                "Mucosal Melanoma: Pathological Evolution, Pathway Dependency and Targeted Therapy. Yanni Ma, Ronghui Xia, Xuhui Ma, Robert L Judson-Torres, Hanlin Zeng. Front Oncol. 2021 Jul 19;11:702287. doi: 10.3389/fonc.2021.702287.",
                "Molecular Biomarkers for Melanoma Screening, Diagnosis and Prognosis: Current State and Future Prospects. Dekker C Deacon, Eric A Smith, Robert L Judson-Torres. Front Med (Lausanne). 2021 Apr 16;8:642380. doi: 10.3389/fmed.2021.642380. eCollection 2021.",
                "Prognostic Gene Expression Profiling in Cutaneous Melanoma: Identifying the Knowledge Gaps and Assessing the Clinical Benefit. Grossman D, Okwundu N, Bartlett EK, Marchetti MA, Othus M, Coit DG, Hartman RI, Leachman SA, Berry EG, Korde L, Lee SJ, Bar-Eli M, Berwick M, Bowles T, Buchbinder EI, Burton EM, Chu EY, Curiel-Lewandrowski C, Curtis JA, Daud A, Deacon DC, Ferris LK, Gershenwald JE, Grossmann KF, Hu-Lieskovan S, Hyngstrom J, Jeter JM, Judson-Torres RL, Kendra KL, Kim CC, Kirkwood JM, Lawson DH, Leming PD, Long GV, Marghoob AA, Mehnert JM, Ming ME, Nelson KC, Polsky D, Scolyer RA, Smith EA, Sondak VK, Stark MS, Stein JA, Thompson JA, Thompson JF, Venna SS, Wei ML, Swetter SM.  JAMA Dermatol. 2020 Jul 29;. doi: 10.1001/jamadermatol.2020.1729. [Epub ahead of print] PubMed PMID: 32725204.",
                "The genomic landscapes of individual melanocytes from human skin. Jessica Tang, Eleanor Fewings, Darwin Chang, Hanlin Zeng, Shanshan Liu, Aparna Jorapur, Rachel L. Belote, Andrew S. McNeal, Iwei Yeh, Sarah T. Arron, Robert L. Judson-Torres, Boris C. Bastian, A. Hunter Shain, Nature, 7 Oct 2020",
                "Ciliation Index Is a Useful Diagnostic Tool in Challenging Spitzoid Melanocytic Neoplasms. Lang UE, Torres R, Cheung C, Vladar EK, McCalmont TH, Kim J, Judson-Torres RL.  J Invest Dermatol. 2020 Jul;140(7):1401-1409.e2. doi: 10.1016/j.jid.2019.11.028. Epub 2020 Jan 22. PubMed PMID: 31978411.​",
                "Quantifying the Rate, Degree, and Heterogeneity of Morphological Change during an Epithelial to Mesenchymal Transition Using Digital Holographic Cytometry. Sofia Kamlund, Birgit Janicke , Kersti Alm, Robert L. Judson-Torres and Stina Oredsson Appl. Sci. 2020, 10(14), 4726; https://doi.org/10.3390/app10144726",
                "Label-Free Classification of Apoptosis, Ferroptosis and Necroptosis Using Digital Holographic Cytometry. Kendra L. Barker, Kenneth M. Boucher and Robert L. Judson-Torres Appl. Sci. 2020, 10(13), 4439; https://doi.org/10.3390/app10134439 Received: 26 May 2020 / Revised: 23 June 2020 / Accepted: 25 June 2020 / Published: 27 June 2020",
                "The Evolution of Melanoma - Moving beyond Binary Models of Genetic Progression. Zeng H, Judson-Torres RL, Shain AH.  J Invest Dermatol. 2020 Feb;140(2):291-297. doi: 10.1016/j.jid.2019.08.002. Epub 2019 Oct 14. Review. PubMed PMID: 31623932; PubMed Central PMCID: PMC6983335.",
                "MicroRNA Ratios Distinguish Melanomas from Nevi. Torres R, Lang UE, Hejna M, Shelton SJ, Joseph NM, Shain AH, Yeh I, Wei ML, Oldham MC, Bastian BC, Judson-Torres RL. J Invest Dermatol. 2020 Jan;140(1):164-173.e7. doi: 10.1016/j.jid.2019.06.126. Epub 2019 Sep 30. PubMed PMID: 31580842; PubMed Central PMCID: PMC6926155.",
                "Genetic Heterogeneity of BRAF Fusion Kinases in Melanoma Affects Drug Responses. Botton T, Talevich E, Mishra VK, Zhang T, Shain AH, Berquet C, Gagnon A, Judson RL, Ballotti R, Ribas A, Herlyn M, Rocchi S, Brown KM, Hayward NK, Yeh I, Bastian BC.  Cell Rep. 2019 Oct 15;29(3):573-588.e7. doi: 10.1016/j.celrep.2019.09.009. PubMed PMID: 31618628; PubMed Central PMCID: PMC6939448.",
                "Research Techniques Made Simple: Feature Selection for Biomarker Discovery. Torres R, Judson-Torres RL.  J Invest Dermatol. 2019 Oct;139(10):2068-2074.e1. doi: 10.1016/j.jid.2019.07.682. Review. PubMed PMID: 31543209.",
                "Evaluation of holographic imaging cytometer holomonitor M4® motility applications. Zhang Y, Judson RL.  Cytometry A. 2018 Nov;93(11):1125-1131. doi: 10.1002/cyto.a.23635. Epub 2018 Oct 21. PubMed PMID: 30343513.",
                "Bi-allelic Loss of CDKN2A Initiates Melanoma Invasion via BRN2 Activation. Zeng H, Jorapur A, Shain AH, Lang UE, Torres R, Zhang Y, McNeal AS, Botton T, Lin J, Donne M, Bastian IN, Yu R, North JP, Pincus L, Ruben BS, Joseph NM, Yeh I, Bastian BC, Judson RL.  Cancer Cell. 2018 Jul 9;34(1):56-68.e9. doi: 10.1016/j.ccell.2018.05.014. PubMed PMID: 29990501; PubMed Central PMCID: PMC6084788.",
                "Genomic and Transcriptomic Analysis Reveals Incremental Disruption of Key Signaling Pathways during Melanoma Evolution. Shain AH, Joseph NM, Yu R, Benhamida J, Liu S, Prow T, Ruben B, North J, Pincus L, Yeh I, Judson R, Bastian BC.  Cancer Cell. 2018 Jul 9;34(1):45-55.e4. doi: 10.1016/j.ccell.2018.06.005. PubMed PMID: 29990500; PubMed Central PMCID: PMC6319271.",
                "Combined activation of MAP kinase pathway and β-catenin signaling cause deep penetrating nevi. Yeh I, Lang UE, Durieux E, Tee MK, Jorapur A, Shain AH, Haddad V, Pissaloux D, Chen X, Cerroni L, Judson RL, LeBoit PE, McCalmont TH, Bastian BC, de la Fouchardière A.  Nat Commun. 2017 Sep 21;8(1):644. doi: 10.1038/s41467-017-00758-3. PubMed PMID: 28935960; PubMed Central PMCID: PMC5608693.",
                "High accuracy label-free classification of single-cell kinetic states from holographic cytometry of human melanoma cells. Hejna M, Jorapur A, Song JS, Judson RL.  Sci Rep. 2017 Sep 20;7(1):11943. doi: 10.1038/s41598-017-12165-1. PubMed PMID: 28931937; PubMed Central PMCID: PMC5607248.",
                "CDK1 inhibition targets the p53-NOXA-MCL1 axis, selectively kills embryonic stem cells, and prevents teratoma formation. Huskey NE, Guo T, Evason KJ, Momcilovic O, Pardo D, Creasman KJ, Judson RL, Blelloch R, Oakes SA, Hebrok M, Goga A.  Stem Cell Reports. 2015 Mar 10;4(3):374-89. doi: 10.1016/j.stemcr.2015.01.019. Epub 2015 Feb 26. PubMed PMID: 25733019; PubMed Central PMCID: PMC4375943.",
                "Two miRNA clusters reveal alternative paths in late-stage reprogramming. Parchem RJ, Ye J, Judson RL, LaRussa MF, Krishnakumar R, Blelloch A, Oldham MC, Blelloch R.  Cell Stem Cell. 2014 May 1;14(5):617-31. doi: 10.1016/j.stem.2014.01.021. Epub 2014 Mar 13. PubMed PMID: 24630794; PubMed Central PMCID: PMC4305531.",
                "MicroRNA-based discovery of barriers to dedifferentiation of fibroblasts to pluripotent stem cells. Judson RL, Greve TS, Parchem RJ, Blelloch R.  Nat Struct Mol Biol. 2013 Oct;20(10):1227-35. doi: 10.1038/nsmb.2665. Epub 2013 Sep 15. PubMed PMID: 24037508; PubMed Central PMCID: PMC3955211.",
                "microRNA control of mouse and human pluripotent stem cell behavior. Greve TS, Judson RL, Blelloch R.  Annu Rev Cell Dev Biol. 2013;29:213-239. doi: 10.1146/annurev-cellbio-101512-122343. Epub 2013 Jul 12. Review. PubMed PMID: 23875649; PubMed Central PMCID: PMC4793955.",
                "Multiple targets of miR-302 and miR-372 promote reprogramming of human fibroblasts to induced pluripotent stem cells. Subramanyam D, Lamouille S, Judson RL, Liu JY, Bucay N, Derynck R, Blelloch R.  Nat Biotechnol. 2011 May;29(5):443-8. doi: 10.1038/nbt.1862. Epub 2011 Apr 13. PubMed PMID: 21490602; PubMed Central PMCID: PMC3685579.",
                "miR-380-5p represses p53 to control cellular survival and is associated with poor outcome in MYCN-amplified neuroblastoma. Swarbrick A, Woods SL, Shaw A, Balakrishnan A, Phua Y, Nguyen A, Chanthery Y, Lim L, Ashton LJ, Judson RL, Huskey N, Blelloch R, Haber M, Norris MD, Lengyel P, Hackett CS, Preiss T, Chetcuti A, Sullivan CS, Marcusson EG, Weiss W, L'Etoile N, Goga A.  Nat Med. 2010 Oct;16(10):1134-40. doi: 10.1038/nm.2227. Epub 2010 Sep 26. PubMed PMID: 20871609; PubMed Central PMCID: PMC3019350.",
                "Opposing microRNA families regulate self-renewal in mouse embryonic stem cells. Melton C, Judson RL, Blelloch R.  Nature. 2010 Feb 4;463(7281):621-6. doi: 10.1038/nature08725. Epub 2010 Jan 6. PubMed PMID: 20054295; PubMed Central PMCID: PMC2894702.",
                "Embryonic stem cell-specific microRNAs promote induced pluripotency. Judson RL, Babiarz JE, Venere M, Blelloch R.  Nat Biotechnol. 2009 May;27(5):459-61. doi: 10.1038/nbt.1535. Epub 2009 Apr 12. PubMed PMID: 19363475; PubMed Central PMCID: PMC2743930.",
                "The GP(Y/F) domain of TF1 integrase multimerizes when present in a fragment, and substitutions in this domain reduce enzymatic activity of the full-length protein. Ebina H, Chatterjee AG, Judson RL, Levin HL.  J Biol Chem. 2008 Jun 6;283(23):15965-74. doi: 10.1074/jbc.M801354200. Epub 2008 Apr 8. PubMed PMID: 18397885; PubMed Central PMCID: PMC2414268.",
                "The self primer of the long terminal repeat retrotransposon Tf1 is not removed during reverse transcription. Atwood-Moore A, Yan K, Judson RL, Levin HL. J Virol. 2006 Aug;80(16):8267-70. doi: 10.1128/JVI.01915-05. PubMed PMID: 16873283; PubMed Central PMCID: PMC1563812."]
            path = os.path.join(data_path, "torres.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for p in publications:
                    file.write(str(p) + "\n")
        
        if lab == "myers":
            div = soup.find('div', class_='paragraph')
            texts = []
            for tag in div.find_all('li'):
                text = tag.text.split("pdf")[0].strip()
                text = text.replace('\u200b', '').replace('\xa0', '')
                if text.split(" ")[0] != "Commentary":
                    texts.append(text)
            path = os.path.join(data_path, "myers.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")

        if lab == "tan":
            heading = soup.find('h3', string="\nPAPERS\n")
            ol_tags = heading.find_all_next('ol')
            texts = []
            for ol in ol_tags:
                for li in ol.find_all('li'):
                    text = li.text.strip()
                    text = text.replace("\n", "").replace("[PDF]", "")
                    texts.append(text)
            path = os.path.join(data_path, "tan.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")

        if (lab == "welm-a") or (lab == "welm-b"):
            divs = soup.find_all('div', class_='docsum-content')
            texts = []
            for div in divs:
                text = div.get_text(strip=True).replace("Free PMC article.", "").replace("Free article.", "").strip()
                texts.append(text)
            path = os.path.join(data_path, "{}.txt".format(lab))
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")

        if lab == "wu":
            div = soup.find_all('div', class_='coh-wysiwyg')[1]
            texts = []
            for p in div.find_all('p'):
                text = p.text.replace("\n", " ").strip()
                texts.append(text)
            path = os.path.join(data_path, "wu.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")
        
        if lab == "ulrich":
            divs = soup.find_all('div', class_='coh-wysiwyg')[1:]
            texts = []
            for div in divs:
                for p in div.find_all('p'):
                    text = p.text.replace("\n", " ").strip()
                    texts.append(text)
            path = os.path.join(data_path, "ulrich.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text)+"\n")

class PubMedBertBaseEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # generate text embeddings
        tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
        inputs = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            output = model(**inputs)
        embeddings = self.meanpooling(output, inputs['attention_mask'])
        return embeddings.tolist()
    
    def meanpooling(self, output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)