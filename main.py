from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from langchain_ollama import ChatOllama

load_dotenv()
def main():
    print("Hello from langchain-course!")
    information = """
Shivaji Rao Gaikwad[a][4] (born 12 December 1950), known professionally as Rajinikanth,[b] is an Indian actor who predominantly works in Tamil cinema.[6] In a career spanning over five decades, he has done 170 films[c] that includes films in Tamil, Hindi, Telugu, Kannada, Bangla, and Malayalam.[7] He is widely regarded to be one of the most successful and popular actors in the history of Indian cinema.[8][9] Known for his uniquely styled mannerism and one liners in films, he has a huge fan base and a cult following. The Government of India honoured him with the Padma Bhushan in 2000 and the Padma Vibhushan in 2016, India's third and second highest civilian honours respectively, and the Dadasaheb Phalke Award in 2019, the highest Indian award in the field of cinema, for his contributions to Indian cinema.[10][11] He has won numerous film awards including one National Film Award, seven Tamil Nadu State Film Awards, a Nandi Award, one Filmfare Award and two Maharashtra State Film Awards.

Following his debut in K. Balachander's 1975 Tamil drama Apoorva Raagangal, Rajinikanth's acting career commenced with a brief phase of portraying antagonistic characters in Tamil films. His major positive role as a scorned lover in S. P. Muthuraman's Bhuvana Oru Kelvi Kuri (1977), 1978's Mullum Malarum and Aval Appadithan received him critical acclaim; the former earned him a Tamil Nadu State Film Award Special Prize for Best Actor.[12][13] By the end of the decade, he had worked in all South Indian film industries and established a career in Tamil cinema. He then played dual roles in the action thriller Billa (1980), a remake of the Hindi film Don (1978). It was his biggest commercial success to that point, earned him stardom and gave him the action hero image.[14] He starred in triple role in Moondru Mugam (1982), which earned him a special prize at the Tamil Nadu State Film Awards ceremony. The following year, he made his Hindi film debut with T. Rama Rao's top grossing Andhaa Kaanoon (1983).[15] Nallavanukku Nallavan (1984) won him that year's Filmfare Award for Best Tamil Actor.[16] In the latter half of the 1980s, he starred in several successful films in Tamil and Hindi, including Geraftaar (1985), Padikkadavan (1985), Mr. Bharath (1986), Dosti Dushmani (1986), Velaikaran (1987), Manithan (1987), Dharmathin Thalaivan (1988) and ChaalBaaz (1989).[17][18][19]

In 1991, Mani Ratnam's Tamil crime film Thalapathi, earned him major critical acclaim for his performance.[20] He collaborated with Suresh Krissna for many films including Annaamalai (1992) and Baashha (1995); the latter was the biggest commercial success in his career yet as well as the highest-grossing film in Tamil for many years.[21] His other success includes P. Vasu's Mannan (1992), Uzhaippali (1993) and K. S. Ravikumar's Muthu (1995) and Padayappa (1999); the latter, which went on to become his and Tamil cinema's highest-grossing movie, exceeding Baashha.[22]

After a few years of hiatus, he returned to acting with the comedy horror film Chandramukhi (2005); it went on to become again the highest-grossing Tamil film. His next, S. Shankar's Sivaji (2007) was the third Indian film and the first ever Tamil film to enter the 100 Crore Club. He then played dual role as a scientist and an andro-humanoid robot in the science fiction film Enthiran (2010) and its sequel 2.0 (2018), both being India's most expensive productions at the time of their release and among the highest-grossing Indian films of all time.[d] In 2023, his blockbuster Jailer made a significant impact in the Tamil film industry, earning over ₹600 crore (about US$70M). Later, in 2025, his film Coolie crossed ₹500 crore, making him the only actor in the industry with three films surpassing the ₹500 crore mark.[24]

Rajinikanth was also named one of the most influential persons in South Asia by Asiaweek.[25][26] He was also named by Forbes India as the most influential Indian of the year 2010.[27]

In 2024, Rajinikanth received the UAE Golden Visa.[28]

Early life and background
Rajinikanth was born as Shivaji Rao Gaikwad on 12 December 1950 in a Marathi family in Bangalore, Mysore State (present day Karnataka).[29][1][30] His mother was a homemaker,[e] and his father Ramoji Rao Gaekwad was a police constable.[1] His ancestors hailed from Mavadi Kadepathar, Pune district, Maharashtra.[32][33] He is the youngest of four siblings in a family consisting of two elder brothers (Satyanarayana Rao and Nageshwara Rao) and a sister (Aswath Balubhai).[34][29] After his father's retirement from work in 1956, the family moved to the suburb of Hanumantha Nagar in Bangalore and built a house there.[29] He lost his mother at the age of nine.[35]

Rajinikanth had his primary education at the Gavipuram Government Kannada Model Primary School in Bangalore.[36] As a child, he was "studious and mischievous" with a great interest in cricket, football and basketball. During this time, his brother enrolled him at the Ramakrishna Math, a Hindu monastery set up by the Ramakrishna Mission. In the math, he was taught Vedas, tradition and history, which eventually instilled a sense of spirituality in him.[37] In addition to spiritual lessons, he also began acting in plays at the math. His aspiration towards theatre grew at the math and was once given an opportunity to enact the role of Ekalavya's friend from the Hindu epic Mahabharata. His performance in the play received praise from the Kannada poet D. R. Bendre.[29] After sixth grade, Rajinikanth was enrolled at the Acharya Pathasala Public School and studied there till completion of his pre-university course.[37] During his schooling at the Acharya Pathasala, he spent a lot of time acting in plays.
Upon completion of his school education, Rajinikanth performed several jobs including that of a coolie,[38] before getting a job in the Bangalore Transport Service as a bus conductor.[39][40] He continued to take part in plays after the Kannada playwright Topi Muniappa offered him a chance to act in one of his mythological plays. He decided to take up an acting course in the newly formed Madras Film Institute after coming across an advertisement.[41] Although his family was not fully supportive of his decision to join the institute,[41] his friend and co-worker Raj Bahadur motivated him to join the institute and financially supported him during this phase.[42][43] During his stay at the institute, he was noticed by the Tamil film director K. Balachander.[44] Balachander provided Rajinikanth with his stage name to avoid confusion with fellow actor Sivaji Ganesan, having taken it from a character's name in his earlier film Major Chandrakanth.[45][46] The director advised him to learn to speak Tamil, a recommendation that Rajinikanth quickly followed.[47] Although he can read the language, he cannot write in it.[48]

    """

    summary_template ="""
        given the information {information} about the   person i want you to create:
        1. A short summary
        2. two interesting facts about them      
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm = ChatOllama(temperature =0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.content)

if __name__ == "__main__":
    main()
