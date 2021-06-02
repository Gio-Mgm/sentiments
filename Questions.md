## Qu'est ce que HTTP?

Hypertext Transfer Protocol (littéralement « protocole de transfert hypertexte ») est un protocole de communication client-serveur développé pour le World Wide Web. HTTPS (avec S pour secured, soit « sécurisé ») est la variante sécurisée par l'usage des protocoles Transport Layer Security (TLS).

## Qu'est ce qu'une API?

Application Programming Interface (interface de programmation applicative) est un ensemble normalisé de classes, de méthodes, de fonctions et de constantes qui sert de façade par laquelle un logiciel offre des services à d'autres logiciels. 

## Quelle est la spécificité des API REST?

REST (pour REpresentational State Transfer) est une type d’architecture d’API qui fournit un certain nombre de normes et de conventions à respecter pour faciliter la communication entre applications. Les APIs qui respectent le standard REST sont appelées API REST ou API RESTful.

Le standard REST impose six contraintes architecturales qui doivent toutes être respectées par un système pour qu’il soit qualifiable de système RESTful. Le strict respect de ces six contraintes permet d’assurer une fiabilité, une scalabilité et une extensibilité optimales.

1. La séparation entre client et serveur 
2. L’absence d’état de sessions
3. L’uniformité de l’interface
4. La mise en cache
5. L’architecture en couches
6. Le code à la demande

## Qu'est ce qu'un URI, un endpoint, une opération?

- Un URI, de l'anglais Uniform Resource Identifier, est une courte chaîne de caractères identifiant une ressource sur un réseau (par exemple une ressource Web) physique ou abstraite, et dont la syntaxe respecte une norme d'Internet mise en place pour le World Wide Web 

- Un Endpoint est ce qu'on appelle une extrémité d'un canal de communication. Autrement dit, lorsqu'une API interagit avec un autre système, les points de contact de cette communication sont considérés comme des Endpoints. Ainsi, pour les API, un Endpoint peut inclure une URL d'un serveur ou d'un service.

- Les opérations peuvent être appliquées à une ressource exposée par l'API. Du point de vue de la mise en œuvre, une opération est un lien entre une ressource, une route et son contrôleur associé.
Si aucune opération n'est spécifiée, toutes les opérations CRUD par défaut (Create, Read, Update, Delete) sont automatiquement enregistrées. Il est également possible - et recommandé pour les grands projets - de définir explicitement les opérations.

## Que trouve-t-on dans la documentation d'une API rest?

- comment appeler l'API.
- les différentes requêtes que l'on peut effectuer, et les paramêtres associés 


## Utiliser Postman pour faire une 3 requêtes sur l'API publique de votre choix. Partagez les requètes ainsi que les réponses.

API utilisé :

![The Internet Chuck Norris Database](http://icndb.com/wp-content/uploads/2011/01/icndb_logo2.png)


request : 

```
http://api.icndb.com/jokes/random/
```

response :

```json
{
    "type": "success",
    "value": {
        "id": 6,
        "joke": "Since 1940, the year Chuck Norris was born, roundhouse kick related deaths have increased 13,000 percent.",
        "categories": []
    }
}
```
### Fetching the list of joke categories

request : 

```
http://api.icndb.com/categories
```

response :


```json
{
    "type": "success",
    "value": [
        "explicit",
        "nerdy"
    ]
}
```

### Fetching the number of jokes

request : 

```
http://api.icndb.com/jokes/count
```

response :


```json
{
    "type": "success",
    "value": 574
}
```