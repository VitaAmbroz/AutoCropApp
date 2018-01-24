# Webová aplikace pro automatický ořez fotografií

Prozatím konzolová aplikace C++, ve které jsou implementovány vybrané metody automatického ořezu obrazu. V současné chvíli je postupně realizována implementace jednotlivých částí metod popsaných v těchto článcích [1,2].

    [1] Stentiford, F. : Attention Based Auto Image Cropping.
    In Workshop on Computational Attention and Applications (WCAA 2007), 2007.
    
    [2] Fang, C.; Lin, Z.; Mech, R.; aj.
    Automatic Image Cropping Using Visual Composition, Boundary Simplicity and Content Preservation Models.
    In Proceedings of the 22Nd ACM International Conference on Multimedia, 2014.

Aplikace používá tyto nástroje a knihovny, je tedy nezbytné je mít správně nainstalovány.
* <a href=https://cmake.org>CMake</a> - `sudo apt-get install cmake`
* <a href=https://opencv.org>OpenCV</a>
* <a href=http://www.vlfeat.org>VlFeat</a>

## Sestavení
Pro sestavení je použit nástroj CMake. V souboru CMakeLists.txt jsou definována pravidla pro vytvoření souborů potřebných k překladu a sestavení. Je zde nutné upravit cestu pro použití knihovny VLFeat, tedy nastavit ji na adresář, kde jsou soubory, které vzniknou po rozbalení této knihovny (více na http://www.vlfeat.org)

Sestavení lze po stažení tohoto repozitáře dosáhnout například takto:

    $ cmake .
    $ make

## Spuštění a popis aplikace
Po úspěšném překladu je možné aplikaci spustit v tomto formátu, kde 'autocrop' je název spustitelného souboru a 'imagePath' je soubor, respektive cesta k souboru, pro který má být proveden automatický ořez:

    $ ./autocrop imagePath

Velikost a další podmínky výstupního ořezu není možné definovat formou argumentů. Lze je manuálně změnit přímo v souboru main.cpp. Nejprve jsou provedeny výsledky ořezu první metodou [1] a poté pro druhou metodu [2]. Postupně se budou v nových oknech zobrazovat obrázky a vypisovat informace do konzole. Po každém otevřeném okně je třeba stisknout klávesu, nebo případě odebrat cv::waitKey(0) v main.cpp.
Implementace vytvoření saliency map a algoritmu ořezu zatím nejsou zoptimalizovány a jsou výrazně pomalejší.
