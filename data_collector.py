"""
Data Collection Module for IIT Jodhpur Word2Vec Assignment
This script scrapes textual data from various IIT Jodhpur sources.
We're targeting: official website, academic regulations, faculty pages, etc.

Author: Bhargav Shekokar (B23CS1008)
Course: NLU Assignment 2
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from tqdm import tqdm
import PyPDF2
from io import BytesIO

# okay so the idea is to collect text from multiple sources
# IIT Jodhpur has a lot of public pages we can scrape

class IITJDataCollector:
    """
    Collects text data from IIT Jodhpur's official sources.
    Basically a web scraper but focused on educational content.
    """
    
    def __init__(self, output_dir="./data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # some common headers to avoid getting blocked
        # websites usually don't like bots lol
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # list of pages we want to scrape from IIT Jodhpur
        # focusing on academic and research related pages
        self.target_urls = {
            'main_pages': [
                'https://iitj.ac.in/',
                'https://iitj.ac.in/about/',
                'https://iitj.ac.in/academics/',
                'https://iitj.ac.in/research/',
                'https://iitj.ac.in/administration/',
            ],
            'departments': [
                'https://cse.iitj.ac.in/',
                'https://ee.iitj.ac.in/',
                'https://me.iitj.ac.in/',
                'https://civil.iitj.ac.in/',
                'https://che.iitj.ac.in/',
                'https://maths.iitj.ac.in/',
                'https://physics.iitj.ac.in/',
                'https://chem.iitj.ac.in/',
                'https://biology.iitj.ac.in/',
                'https://hss.iitj.ac.in/',
            ],
            'academic_programs': [
                'https://iitj.ac.in/academics/index.php?id=academic_programmes',
                'https://iitj.ac.in/academics/index.php?id=btech',
                'https://iitj.ac.in/academics/index.php?id=mtech',
                'https://iitj.ac.in/academics/index.php?id=phd',
                'https://iitj.ac.in/academics/index.php?id=msc',
            ],
        }
        
        # collected texts will be stored here
        self.collected_texts = []
        
    def fetch_page(self, url, retries=3):
        """
        Fetches a webpage and returns soup object.
        Has retry logic because networks can be flaky sometimes.
        """
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'lxml')
                else:
                    print(f"Got status {response.status_code} for {url}")
            except requests.RequestException as e:
                print(f"Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(2)  # wait a bit before retrying
        return None
    
    def extract_text_from_html(self, soup):
        """
        Extracts clean text from BeautifulSoup object.
        Removes all the HTML garbage and keeps only the good stuff.
        """
        if not soup:
            return ""
        
        # get rid of script and style elements - they're useless for us
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # also remove these common non-content elements
        for element in soup.find_all(class_=re.compile(
            r'nav|menu|sidebar|footer|header|advertisement|social', re.I)):
            element.decompose()
        
        # get text and clean it up
        text = soup.get_text(separator=' ')
        
        # clean up whitespace - websites have so much unnecessary whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def scrape_all_websites(self):
        """
        Main scraping function - goes through all our target URLs
        and collects text from them.
        """
        print("Starting web scraping...")
        print("=" * 50)
        
        all_urls = []
        for category, urls in self.target_urls.items():
            all_urls.extend([(category, url) for url in urls])
        
        for category, url in tqdm(all_urls, desc="Scraping pages"):
            soup = self.fetch_page(url)
            if soup:
                text = self.extract_text_from_html(soup)
                if len(text) > 100:  # only keep if we got meaningful content
                    self.collected_texts.append({
                        'source': url,
                        'category': category,
                        'text': text
                    })
                    print(f"✓ Got {len(text)} chars from {url[:50]}...")
            
            # be nice to the server - don't hammer it
            time.sleep(1)
        
        print(f"\nCollected text from {len(self.collected_texts)} pages")
        return self.collected_texts
    
    def download_pdf(self, url, filename):
        """
        Downloads a PDF and extracts text from it.
        PDFs are tricky - some work, some don't.
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                pdf_file = BytesIO(response.content)
                reader = PyPDF2.PdfReader(pdf_file)
                
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + " "
                
                return text.strip()
        except Exception as e:
            print(f"Failed to process PDF {url}: {e}")
        return ""
    
    def collect_academic_regulations(self):
        """
        Downloads and processes academic regulation documents.
        These are usually PDFs so we need special handling.
        """
        print("\nLooking for academic regulation documents...")
        
        # known academic regulation PDF URLs (these might change over time)
        # update these if they don't work
        regulation_urls = [
            'https://iitj.ac.in/uploaded_docs/academics/UG_Academic_Regulations.pdf',
            'https://iitj.ac.in/uploaded_docs/academics/PG_Academic_Regulations.pdf',
            'https://iitj.ac.in/uploaded_docs/academics/PhD_Regulations.pdf',
        ]
        
        for url in regulation_urls:
            text = self.download_pdf(url, url.split('/')[-1])
            if text:
                self.collected_texts.append({
                    'source': url,
                    'category': 'academic_regulations',
                    'text': text
                })
                print(f"✓ Extracted text from {url.split('/')[-1]}")
    
    def save_raw_data(self):
        """
        Saves all collected data to files.
        Each source gets its own file + one combined file.
        """
        # save individual files
        for i, item in enumerate(self.collected_texts):
            filename = f"doc_{i:03d}_{item['category']}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {item['source']}\n")
                f.write(f"Category: {item['category']}\n")
                f.write("-" * 50 + "\n")
                f.write(item['text'])
        
        # also create one big combined file
        combined_path = os.path.join(self.output_dir, "combined_corpus.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            for item in self.collected_texts:
                f.write(item['text'] + "\n\n")
        
        print(f"\nSaved {len(self.collected_texts)} documents to {self.output_dir}")
        print(f"Combined corpus saved to {combined_path}")
        
        return combined_path


def create_sample_corpus():
    """
    Creates a sample corpus with IIT Jodhpur related content.
    This is useful if web scraping fails or takes too long.
    The content is based on publicly available information about IIT Jodhpur.
    
    Note: I've included common academic terms multiple times so they appear
    in the vocabulary after min_count filtering.
    """
    
    sample_texts = [
        # Academic Programs - expanded with UG/PG terms
        """
        Indian Institute of Technology Jodhpur offers various undergraduate and postgraduate programs.
        The BTech program is a four year UG undergraduate course offered in multiple disciplines.
        Students can pursue BTech in Computer Science and Engineering, Electrical Engineering,
        Mechanical Engineering, Civil Engineering, Chemical Engineering, Bioscience and Bioengineering,
        and Artificial Intelligence and Data Science. UG students learn fundamental concepts in their
        first two years. The MTech program is a two year PG postgraduate course focusing on advanced
        research and specialized knowledge. PG students learn specialized topics and conduct research.
        PhD programs at IIT Jodhpur are research intensive and span three to five years. The institute
        also offers MSc programs in Physics, Chemistry, and Mathematics. Dual degree programs combining
        BTech and MTech are available for exceptional students. UG and PG academic curriculum emphasizes
        both theoretical foundations and practical applications. Students learn through lectures,
        tutorials, and laboratory sessions.
        """,
        
        # Research at IIT Jodhpur - expanded
        """
        Research at IIT Jodhpur spans multiple domains including artificial intelligence, machine learning,
        renewable energy, smart materials, and computational sciences. Faculty members actively publish
        in top tier journals and conferences. Professor and faculty work together on cutting edge research.
        The institute has established several research centers of excellence. Sponsored research projects
        from government agencies and industry partners contribute to the research ecosystem. PhD students
        work on cutting edge problems under faculty supervision. Every professor guides multiple research
        students. Research collaboration with international universities enhances the global exposure.
        The Technology Innovation Hub focuses on smart healthcare and advanced sensors. Conference
        presentations and journal publications are encouraged for all research students. Patent filing
        and technology transfer activities are growing each year. Faculty and professor positions attract
        top talent from around the world.
        """,
        
        # Student Life - expanded with hostel and professor
        """
        Student life at IIT Jodhpur is vibrant and engaging. The campus hosts various cultural festivals
        and technical events throughout the year. Ignus is the annual cultural festival that attracts
        participants from across the country. Varchas is the sports festival promoting physical fitness
        and competitive spirit. Prometeo is the technical festival where students showcase innovations.
        Hostel provides comfortable accommodation with modern amenities for all students. Each hostel
        has a warden who is usually a professor. The student council organizes various activities and
        represents student interests. Clubs and societies cover diverse interests from robotics to
        photography. Students live in hostel during their entire program duration. Hostel life teaches
        students independence and time management. Professor mentors guide students in academic matters.
        Placement season sees top companies recruiting students for excellent opportunities. Internships
        during summer provide industry exposure. Alumni network supports current students through
        mentorship programs. Hostel mess provides nutritious food to students.
        """,
        
        # Examination and Evaluation - expanded with exam
        """
        The examination system at IIT Jodhpur follows a continuous evaluation approach. Mid semester
        exam tests student understanding of first half of course. End semester exam is a comprehensive
        assessment of entire course material. Quiz exam and assignment marks contribute to the final
        grade. Laboratory courses include practical exam and viva voce. Projects require presentations
        and demonstrations. The grading system uses letter grades from A to F based on exam performance.
        Grade Point Average is calculated from all exam scores. Students must maintain minimum CGPA to
        continue in the program. Supplementary exam is available for failed courses. Academic probation
        applies when exam performance falls below standards. Course registration happens at the start
        of each semester before exam schedule is released. Exam hall allocation is done by examination
        section. Students learn exam preparation strategies from seniors. Each exam tests different
        aspects of student learning and understanding.
        """,
        
        # Faculty and Departments - expanded with professor
        """
        IIT Jodhpur has distinguished faculty members with expertise in various fields. Each professor
        specializes in specific research areas. The Department of Computer Science and Engineering
        focuses on algorithms, systems, and artificial intelligence. The Department of Electrical
        Engineering covers electronics, communications, and power systems. Professor positions require
        PhD and research experience. Mechanical Engineering department works on design, manufacturing,
        and thermal sciences. Civil and Infrastructure Engineering addresses sustainable construction
        and urban planning. Chemical Engineering focuses on process engineering and materials. The
        Department of Bioscience and Bioengineering explores biotechnology and computational biology.
        Physics department researches condensed matter and photonics. Every professor teaches courses
        and guides research. Chemistry department works on organic synthesis and materials chemistry.
        Mathematics department covers pure and applied mathematics. Humanities and Social Sciences
        department provides liberal arts education. Assistant professor, associate professor, and
        full professor constitute the faculty hierarchy.
        """,
        
        # Academic Regulations - UG expanded
        """
        The UG undergraduate academic regulations govern BTech students at IIT Jodhpur. UG students must
        complete a minimum of 160 credits for graduation. Core courses form the foundation of the UG
        curriculum. Departmental electives allow UG specialization in chosen areas. Open electives
        provide interdisciplinary exposure to UG students. BTP or Bachelor Thesis Project is mandatory
        for UG students in the final year. Summer internship provides industry or research experience
        to UG students. Minimum attendance of 75 percent is required in all UG courses. Course drop
        is allowed within specified deadlines for UG programs. Branch change is possible for UG students
        based on academic performance. Scholarship and fee waiver depend on family income and merit for
        UG admission. Disciplinary action follows plagiarism or academic misconduct by UG students.
        UG students learn fundamental engineering principles. The UG program prepares students for
        industry and higher studies.
        """,
        
        # Academic Regulations - PG and PhD expanded
        """
        PG postgraduate and doctoral programs have specific academic regulations. PG MTech students must
        complete coursework and thesis within two years. Comprehensive exam tests research aptitude of
        PhD students after their PG coursework. Synopsis submission precedes thesis defense for PG and
        PhD programs. External examiner evaluates the doctoral thesis. Publication requirement ensures
        PG research quality. Research supervisor who is a professor guides the PG student throughout
        the program. Progress seminars track research advancement for PG students. Teaching assistantship
        provides financial support to PG students. Research assistantship funds specific PG projects.
        Leave policies cover medical and personal emergencies for PG students. Extension of registration
        requires special approval for PG programs. Thesis plagiarism check is mandatory before PG
        submission. PG students learn advanced research methodologies. The PG program prepares students
        for research careers.
        """,
        
        # Campus and Infrastructure - expanded with hostel
        """
        IIT Jodhpur campus is located in the Karwar region of Rajasthan. The campus spans over 850 acres
        with modern infrastructure. Academic buildings house classrooms and laboratories. Central library
        provides extensive resources and study spaces where students learn. Sports complex includes
        facilities for various indoor and outdoor games. Swimming pool and gymnasium promote fitness.
        Guest house accommodates visitors and parents. Faculty and staff residences are within campus.
        Each professor has an office in the department building. Hostel blocks provide accommodation
        for students. Boys hostel and girls hostel are located in different areas. Each hostel has
        common rooms and recreational facilities. Cafeteria and mess provide food services. Bank and
        ATM facilities are available on campus. Medical centre attends to health needs of hostel
        residents. Solar power and rainwater harvesting support sustainability. WiFi connectivity
        covers the entire campus including hostel areas. Hostel life is an integral part of student
        experience at IIT Jodhpur.
        """,
        
        # Admissions - expanded
        """
        Admission to IIT Jodhpur is highly competitive. JEE Advanced exam is the entrance examination
        for BTech UG programs. GATE exam score is required for MTech PG admission. PhD admission involves
        written exam and interview. International students apply through specific channels. Seat
        allocation happens through centralized counseling after exam results. Document verification
        confirms eligibility after exam clearing. Fee payment confirms admission acceptance. Orientation
        program introduces new UG and PG students to campus life and hostel. Academic calendar specifies
        semester dates and exam schedules. Registration completes the enrollment process. ID card and
        email account are issued to new students. First year students learn about campus facilities.
        UG admission requires strong JEE exam performance. PG admission requires good GATE exam rank.
        Hostel allotment is done during admission process.
        """,
        
        # Placements and Career expanded
        """
        Career Development Cell manages campus placements at IIT Jodhpur. Top companies from IT,
        consulting, finance, and core sectors recruit students. Pre placement talks familiarize
        students with companies. Resume building workshops help students learn professional skills.
        Interview preparation includes mock interviews where students learn communication. Placement
        statistics show excellent records each year. Average package and highest package demonstrate
        placement success. Internship opportunities are facilitated by the placement cell for UG and
        PG students. Industry interaction events bring companies to campus. Alumni referrals support
        job search. Higher studies guidance helps students apply to universities abroad. Professor
        recommendations support student applications. Both UG and PG students participate in placements.
        Students learn about career opportunities through placement cell events. Campus recruitment
        provides job offers before graduation.
        """,
        
        # Teaching and Learning expanded
        """
        Teaching at IIT Jodhpur follows modern pedagogical approaches. Each professor uses innovative
        teaching methods to help students learn effectively. Lectures are complemented by tutorials
        where students learn through problem solving. Laboratory sessions help UG and PG students
        learn practical skills. Students learn from textbooks, online resources, and research papers.
        Professor office hours allow students to learn through one-on-one discussions. Teaching
        assistants help students learn during tutorial sessions. The exam evaluates how well students
        learn the course material. Students learn teamwork through group projects. Presentation skills
        are developed as students learn to communicate technical concepts. Students learn programming,
        mathematics, and domain specific knowledge. The curriculum is designed to help students learn
        both theory and practice. UG students learn foundational concepts while PG students learn
        advanced topics. Students learn from faculty who are experts in their fields.
        """,
        
        # Student academics expanded
        """
        Students at IIT Jodhpur engage in rigorous academic activities. UG students take core courses
        in first two years and specialize later. PG students focus on advanced coursework and research.
        Every semester has mid term exam and end term exam. Students prepare for exam through regular
        study and revision. Hostel study rooms provide quiet spaces for exam preparation. Professor
        consultation helps students understand difficult concepts before exam. Group study in hostel
        improves learning outcomes. Students learn time management to balance coursework and exam
        preparation. Academic counselors guide students facing exam stress. The semester system helps
        students learn systematically. Students learn academic integrity and exam ethics. Project work
        complements exam based evaluation. Students learn research methodology in advanced courses.
        """
    ]
    
    return sample_texts


def collect_data(use_sample=True):
    """
    Main function to collect all the data.
    Set use_sample=True if web scraping isn't working or taking too long.
    """
    
    if use_sample:
        print("Using sample corpus based on IIT Jodhpur content...")
        print("(Set use_sample=False to scrape live data from websites)")
        
        os.makedirs("./data/raw", exist_ok=True)
        
        sample_texts = create_sample_corpus()
        
        # save each document separately
        for i, text in enumerate(sample_texts):
            filename = f"./data/raw/doc_{i:03d}_sample.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text.strip())
        
        # create combined corpus
        combined_path = "./data/raw/combined_corpus.txt"
        with open(combined_path, 'w', encoding='utf-8') as f:
            for text in sample_texts:
                f.write(text.strip() + "\n\n")
        
        print(f"Created {len(sample_texts)} sample documents")
        print(f"Combined corpus saved to {combined_path}")
        
        return combined_path
    
    else:
        # use web scraping
        collector = IITJDataCollector()
        collector.scrape_all_websites()
        collector.collect_academic_regulations()
        return collector.save_raw_data()


if __name__ == "__main__":
    # run this script to collect data
    # change use_sample=False if you want to scrape actual websites
    collect_data(use_sample=True)
