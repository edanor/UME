#pragma once

template <typename SCALAR_T>
struct ScalarToString { static std::string value() { return "Undefined"; } };

template<> struct ScalarToString<uint8_t> { static std::string value() { return "uint8_t"; } };
template<> struct ScalarToString<uint16_t> { static std::string value() { return "uint16_t"; } };
template<> struct ScalarToString<uint32_t> { static std::string value() { return "uint32_t"; } };
template<> struct ScalarToString<uint64_t> { static std::string value() { return "uint64_t"; } };

template<> struct ScalarToString<int8_t> { static std::string value() { return "int8_t"; } };
template<> struct ScalarToString<int16_t> { static std::string value() { return "int16_t"; } };
template<> struct ScalarToString<int32_t> { static std::string value() { return "int32_t"; } };
template<> struct ScalarToString<int64_t> { static std::string value() { return "int64_t"; } };

template<> struct ScalarToString<float> { static std::string value() { return "float"; } };
template<> struct ScalarToString<double> { static std::string value() { return "double"; } };

