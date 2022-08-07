import requests
import time

api = "http://inference-3.tenant-chairesearch-test.knative.chi.coreweave.com/v1/models/model:predict"

data = {
    "instances": ["ear's Fright and tried to kill the night guard, who is Michael, Henry or a random unnamed person. Eventually, the attraction is caught on fire. In the newspaper, Springtrap's head can be seen when brightening up the image, giving an early hint he survived.\n\nIn the opening scene of Sister Location, an entrepreneur is asking him questions about the new animatronics. They inquire why certain features were added and express their concerns, but he avoids answering the specific features they refer to.\n\nHe is also the creator of the Funtime Animatronics (Assisted by an unknowing Henry) and the former owner of the Circus Baby's Entertainment and Rental, and, by extension, Circus Baby's Pizza World.\n\nIt's revealed in the final Michael Afton's Cutscene that William sent his son, Michael, to his rundown factory to find his daughter, but he is 'scooped' as his sister, Baby, tricked him. Ennard took control over his body, but he manages to survive as Michael becomes a rotting corpse. He swears to find him.\n\nWilliam Afton returns as the main antagonist. It's revealed that William's old partner, Henry, lured Springtrap, Scrap Baby (Elizabeth), Molten Freddy (and by extension, the remaining parts of Ennard), and Lefty (the Puppet) to a new Freddy Fazbear's Pizza. Michael in Freddy Fazbear's Pizzeria Simulator is the manager. On Saturday, Henry burns the whole pizzeria down, while he dies in the fire. Michael decides to stay in the fire as well. Springtrap and every other animatronic die in the fire and the souls are free, as their killer is dead.\n\nWhile not directly appearing, footprints that are very similar to Springtrap's can be found behind the house in Midnight Motorist's secret minigame, presumably luring away the child of the abusive father in the game.\n\nSeen when completing the Fruity Maze game, standing next to a girl named Susie from the right is William Afton wearing the Spring Bonnie suit that he eventually was trapped in and became Springtrap he then seemingly murders Susie.\nWilliam Afton: ...\nMe: \u2026\nWilliam Afton:"]
}


for i in range(1):
    start = time.time()
    result = requests.post(api, json=data, timeout=1000000000)
    print(result.text)

