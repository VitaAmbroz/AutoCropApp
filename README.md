# Algoritmy pro automatický ořez fotografií (Algorithms for Automatic Image Cropping)

Konzolová aplikace v C++, ve které jsou implementovány vybrané algoritmy automatického ořezu obrazu. Pro každý z těchto algoritmů byly vytvořeny různé metody způsobu výběru nejlepšího rámečku ořezu. Většinu těchto metod lze spustit z příkazové řádky pomocí příslušných parametrů, případně vlastním experimentováním se zdrojovými kódy.

    [1] Stentiford, F. : Attention Based Auto Image Cropping.
    In ICVS Workshop on Computation Attention and Applications, 2007.
    
    [2] Fang, C.; Lin, Z.; Mech, R.; aj.:
    Automatic Image Cropping Using Visual Composition, Boundary Simplicity and Content Preservation Models.
    In Proceedings of the 22Nd ACM International Conference on Multimedia, 2014.

    [3] Suh, B.; Ling, H.; Bederson, B. B.; aj.:
    Automatic Thumbnail Cropping and its Effectiveness.
    In Proceedings of the 16th Annual ACM Symposium on User Interface Software and Technology, 2003.

Aplikace používá tyto nástroje a knihovny, které je nezbytné mít správně nainstalovány.
* <a href=https://cmake.org>CMake</a> - použita verze 3.11.1
* <a href=https://opencv.org>OpenCV</a> - použita verze 3.4.1
* <a href=http://www.vlfeat.org>VLFeat</a> - použita verze 0.9.21
* <a href=https://www.boost.org/>Boost</a> - použita verze 1.67.0

## Sestavení
Pro sestavení je použit nástroj CMake. V souboru CMakeLists.txt jsou definována pravidla pro vytvoření souborů potřebných k překladu a sestavení. Je zde nutné upravit cestu pro použití knihovny VLFeat, respektive nastavit ji na adresář, kde jsou soubory, které vzniknou po rozbalení této knihovny (více na http://www.vlfeat.org). Pokud bude stažen tento repozitář, stačí pouze rozbalit soubor vlfeat.zip a bude ta dodržena cesta uvedená v souboru CMakeLists.txt.

Sestavení aplikace lze po stažení tohoto repozitáře dosáhnout například takto:

    $ cmake .
    $ make

## Spuštění a popis aplikace
Po úspěšném překladu je možné aplikaci spustit v následujících formátech, kde 'autocrop' je název spustitelného souboru mohou následovat další parametry. Kromě vysvětlení jednotlivých parametrů jsou zde uvedeny i konkrétní příklady spuštění.

Vypíše zprávu, kde jsou uvedeny možné formáty pro spuštění aplikace:
    $ ./autocrop -help

Základní formát, kde 'imagePath' je cesta k originálnímu obrázku, pro který bude proveden automatický ořez(tento parametr je použit ve všech dalších případech):
    $ ./autocrop imagePath (= postupně budou spuštěny všechny algoritmy automatického ořezu [1][2][3])

Výběr algoritmu automatického ořezu, kde parametry reprezentují konkrétní obecný algoritmus (-suh [3], -sten [1], -fang [2]):
    $ ./autocrop imagePath -suh
    $ ./autocrop imagePath -sten
    $ ./autocrop imagePath -fang
    $ ./autocrop imagePath -suh -fang (= postupně se provedou ořezy pomocí algoritmů [3][2])

Výběr metody, která specifikuje výšku a šířku výstupního ořezu(v pixelech):
    $ ./autocrop imagePath -wh 600 400
    $ ./autocrop imagePath -suh -wh 600 400

Výběr metody, která vytvoří ořez, který bude zmenšený oproti originálu v zadaném poměru a bude zachován poměr stran:
    $ ./autocrop imagePath -scale 0.66
    $ ./autocrop imagePath -suh -scale 0.66

Výběr metody, která bude hledat optimální rámeček rámeček na základě zadaného poměru jeho šířky a výšky:
    $ ./autocrop imagePath -whratio 3 2 (= ořez bude ve formátu 3:2)
    $ ./autocrop imagePath -suh -whratio 3 2 (= ořez bude ve formátu 3:2)

Definice prahové hodnoty pro potřebnou míru významu. Je použita pouze ve třetím uvedeném algoritmu ořezu [3]:
    $ ./autocrop imagePath -suh -threshold 0.5
    $ ./autocrop imagePath -suh -whratio 3 2 -threshold 0.5

Vypnutí funkce zobrazování oken. Pokud bude toto zobrazování vypnuto, bude zobrazen vždy pouze originál a výsledné ořezy. Pokud bude zobrazování zapnuto(výchozí), tak budou současně zobrazeny i saliency maps nebo gradient. Při zobrazení nového okna je pro pokračování nutné stistknout libovolnou klávesu.
    $ ./autocrop imagePath -w
    $ ./autocrop imagePath -fang -w

Spuštění tréninku modelu kompozice, který je potřebný pro druhou uvedenou metodu [2]. Parametr 'datasetDir' je cesta k adresáři, kde jsou uloženy obrázky, které budou použité pro trénink:
    $ ./autocrop -train datasetDir


Pokud bude zadána neplatná kombinace parametrů, bude vypsána chybová hláška a program bude ukončen s chybou. Během běhu programu budou do konzole postupně vypisovány informaci o aktuální fázi. Kromě uvedených metod, které je možné spustit z příkazové řádky byla vytvořena i řada dalších způsobů autoamtického ořezu, které lze vyzkoušet manuálním zásahem do implementace.
