query_person = """
                SELECT * WHERE{{
                        <{}> foaf:name ?name;
                             dbo:height ?height;
                             dbo:abstract ?abstract;
                             dbo:birthDate ?dob;
                             dbo:wikiPageID ?wikiID;
                             dbo:birthPlace ?pob.
                OPTIONAL{{<{}> dbo:personFunction ?pf}}
                FILTER (lang(?abstract)='de')
            }}
            """

"""
Query template

SELECT * WHERE
  {
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> foaf:name ?name}
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> dbo:height ?height}
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> dbo:birthDate ?dob}
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> dbo:wikiPageID ?wikiID}
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> dbo:abstract ?abstract}
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> dbo:birthPlace ?pob}
     OPTIONAL{<http://dbpedia.org/resource/Lionel_Messi> dbo:personFunction ?pf}
     FILTER (lang(?abstract)='de' && lang(?name)='en')
}"""