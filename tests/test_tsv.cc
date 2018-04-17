#include <sstream>
#include <string>

#include <catch.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

#include "../tsv/tsv.hpp"


TEST_CASE("load_tsv loads a 5-by-3 tensor")
{
    std::istringstream stream{
        "0  1\t2\n"
        "3\t4  5\n"
        "6  7  8\n"
        "9\t0\t1\n"
        "\t2 3 4\n"
    };

    xt::xtensor<double, 2> const tensor = load_tsv(stream);
    xt::xtensor<double, 2> const expected = {
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8},
        {9, 0, 1},
        {2, 3, 4},
    };

    CHECK(stream.eof());
    CHECK((tensor == expected));
}

TEST_CASE("save_tsv saves a 5-by-3 tensor")
{
    xt::xtensor<double, 2> const tensor = {
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8},
        {9, 0, 1},
        {2, 3, 4},
    };
    std::string const expected =
        "0\t1\t2\n"
        "3\t4\t5\n"
        "6\t7\t8\n"
        "9\t0\t1\n"
        "2\t3\t4\n";

    std::ostringstream stream;
    save_tsv(stream, tensor);

    CHECK(stream.str() == expected);
}
