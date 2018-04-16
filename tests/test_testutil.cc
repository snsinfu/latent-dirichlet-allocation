#include <cmath>
#include <cstddef>

#include <catch.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

#include "testutil.hpp"


TEST_CASE("extract_rows extracts 1-tensor rows from 2-tensor")
{
    xt::xtensor<double, 2> const tensor = {
        {1, 2, 3},
        {4, 5, 6},
    };
    std::vector<xt::xtensor<double, 1>> const rows = extract_rows(tensor);

    CHECK(rows.size() == 2);
    CHECK((rows.at(0) == xt::xtensor<double, 1>{1, 2, 3}));
    CHECK((rows.at(1) == xt::xtensor<double, 1>{4, 5, 6}));
}

TEST_CASE("maxmin_reorder aligns two similar vectors")
{
    std::vector<double> items = {2.1, 4.5, 0.8, 3.2};
    std::vector<int> const reference = {1, 2, 3, 4};

    auto score_similarity = [](double item, int ref) -> double {
        return -std::fabs(item - ref);
    };
    double const min_score = maxmin_reorder(items, reference, score_similarity);

    CHECK(min_score == score_similarity(4.5, 4));

    CHECK(items[0] == 0.8);
    CHECK(items[1] == 2.1);
    CHECK(items[2] == 3.2);
    CHECK(items[3] == 4.5);
}

TEST_CASE("maxmin_reorder works with rvalues")
{
    double const min_score = maxmin_reorder(
        std::vector<double>{2.1, 4.5, 0.8, 3.2},
        std::vector<int>{1, 2, 3, 4},
        [](double item, int ref) { return -std::fabs(item - ref); }
    );
    CHECK(min_score == -0.5);
}
