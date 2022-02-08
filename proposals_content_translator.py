import pandas as pd
import requests

consultation_data = pd.read_csv("consultation_data/rua-publics.csv", encoding="utf-8",  engine='python', quoting=3, sep=';')
proposals = consultation_data.loc[consultation_data["type"] == "opinion"]

# sentences = ["Afin d'être parfaitement équitable, il faut que ce revenu universel le soit. Pour tous, sans aucune condition ni calcul alambiqué. Le travailleur reçoit autant que celui qui ne travaille pas, ainsi il n'y a pas d'hésitation à aller travailler et pas de problèmes d'administration de l'aide ni de fraude.",
#              "Un RUA , donnant droit à un revenu au dessus du seuil de pauvreté ,en échange d'heures d'utilité publique (à definir ) à sa collectivité qui le demandera ou pas ,en fonction des capacités de chacun . ",
#              'Toutes les prestations sociales doivent être récupérables sur la succession.',
#              "ASPA est aujourd'hui récupérable sur succession, ce qui n'est pas le cas des autres prestations que le gouvernement souhaite fusionner dans ce nouveau dispositif. Il ne faudrait pas que le revenu universel d'activité soit récupérable sur succession si l'ASPA y est intégrée.",
#              "Un système simple, universel, individuel, garanti, sans bureaucratie et calculs complexes. Example : 600 euros par adultes et 300 euros par mineur (-18ans) à charge en dessous de 18ans versé au parents chaque mois. Réellement universel : versé quelque soit le revenu de l’individu et de sa composition familiale. Si on perd son emploi, si une relation de couple se termine, il n’y a pas de calculs à faire, le système marche toujours de la même manière. Un système égalitaire : le montant sera versé à tou(te)s sans besoin d’en faire la demande, car le Revenu Universel (RU) sera versé comme surplus d’impôt. Les individus avec des moyens plus élevés ne recevrons pas le RU car s’ils/elles payent l’impôt sur le revenu, le RU sera déduis automatiquement de leurs contributions ; le RU élèvera donc leur seuil minimum d’imposition. Le montant évoluera chaque année avec l’inflation, remplacera presque toutes les allocations existantes (AF, RSA, ASS, ASV). Les allocations d’aide à l’handicap et vieillesse seront transformé en bonus supplémentaire versé en plus du RU, mais attribuées sous réserve de respecter des critères spécifiques. Les jeunes adultes a charges de leurs parents verrons leurs contributions versées à leurs parents jusqu’à un certain âge. Le RU pourra aussi couvrir certaines aides destinées aux étudiants.",
#              "Le jour des 18 ans, le versement automatique débute pour tous sans autre forme de procès. Toute relation avec un niveau de revenu sera dés-incitative à la reprise/augmentation d'activité.",
#              "en présentant des justificatifs de faibles ressources ne dépassant un certain seuil, l'on attribuera alors après étude du dossier le RUA, en se méfiant des fraudeurs qui seront vite tentés de profiter du système alors que ceux qui en ont réellement besoin pourraient ne pas en profiter... "]

sentences = proposals["contributions_bodyText"].tolist()

with open("results/translated-publics-contents.txt", "w", encoding="utf8") as translated:
    for sentence in sentences:
        try:
            if sentence is not None and sentence != "":
                response = requests.get(url="https://api-free.deepl.com/v2/translate", params={
                    "auth_key": "ba067835-2e79-2967-d68a-21aa42ace39d:fx",
                    "text": sentence,
                    "source_lang": "FR",
                    "target_lang": "EN"
                })
                response_sentence = response.json()["translations"][0]["text"]
                translated.write(response_sentence + "\n")
        except Exception as e:
            print(e)
            break
