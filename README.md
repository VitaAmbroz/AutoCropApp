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
Po úspěšném překladu je možné aplikaci spustit v tomto formátu, kde 'autocrop' je název spustitelného souboru a 'imagePath' je soubor, respektive cesta k souboru, pro který má být proveden automatický ořez:

    $ ./autocrop imagePath

Velikost a další podmínky výstupního ořezu není možné definovat formou argumentů. Lze je manuálně změnit přímo v souboru main.cpp. Nejprve jsou provedeny výsledky ořezu první metodou [1] a poté pro druhou metodu [2]. Postupně se budou v nových oknech zobrazovat obrázky a vypisovat informace do konzole. Po každém otevřeném okně je třeba stisknout klávesu, nebo případě odebrat cv::waitKey(0) v main.cpp.

Implementace vytvoření saliency map a algoritmu ořezu zatím nejsou zoptimalizovány a jsou výrazně pomalejší.
