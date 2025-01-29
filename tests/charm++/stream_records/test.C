#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

std::string generateRandomString(int length) {
    std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string result;
    for (int i = 0; i < length; ++i) {
        result += characters[rand() % characters.length()];
    }
    return result;
}

std::vector<std::string> generateStrings(int numStrings, int minLength, int maxLength) {
    std::vector<std::string> strings;
    for (int i = 0; i < numStrings; ++i) {
        int randomLength = minLength + rand() % (maxLength - minLength + 1);
        strings.push_back(generateRandomString(randomLength));
    }
    return strings;
}

int main() {
    srand(static_cast<unsigned int>(time(0)));
    int numStrings, minLength, maxLength;
    std::cout << "num strings: ";
    std::cin >> numStrings;
    std::cout << "min len: ";
    std::cin >> minLength;
    std::cout << "max len: ";
    std::cin >> maxLength;

    std::vector<std::string> randomStrings = generateStrings(numStrings, minLength, maxLength);
    
    std::cout << "\nGenerated Strings:\n";
    for (const std::string &str : randomStrings) {
        std::cout << str << std::endl;
    }


    return 0;
}
