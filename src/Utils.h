#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
// #include <numbers>

namespace constants {
    constexpr double pi = 3.14159265358979323846;
    constexpr double e = 2.71828182845904523536;
    constexpr double inv_pi = 0.31830988618379067154;
    constexpr double inv_sqrtpi = 0.56418958354775628695;
    constexpr double ln2 = 0.69314718055994530942;
    constexpr double ln10 = 2.30258509299404568402;
    constexpr double sqrt2 = 1.41421356237309504880;
    constexpr double sqrt3 = 1.73205080756887729353;
    constexpr double inv_sqrt3 = 0.57735026918962576451;
    constexpr double egamma = 0.57721566490153286060; // Euler-Mascheroni constant
    constexpr double phi = 1.61803398874989484820;    // Golden ratio
}

// #include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

inline std::string doubleToString(double d) {
    std::ostringstream ss;
    ss << std::fixed << d;
    std::string str = ss.str();

    // remove trailing zeros
    size_t lastNonZero = str.find_last_not_of('0');
    if (lastNonZero != std::string::npos && str[lastNonZero] == '.') {
        lastNonZero++;
    }
    str = str.substr(0, lastNonZero + 1);
    return str;
}

/** Returns the given string without the wrapping character */
inline std::string unwrap(const std::string& str, char c = ' ') {
    if (str.front() == c && str.back() == c) {
        return str.substr(1, str.size() - 2);
    }
    return str;
}

inline bool isBlank(const std::string& str) {
    return str.find_first_not_of(' ') == std::string::npos;
}

inline void printFileContents(const std::string& filename, std::ostream& os = std::cout) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            os << line << "\n";
        }
        file.close();
    }
}

#endif //UTILS_H