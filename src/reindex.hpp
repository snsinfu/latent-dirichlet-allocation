#ifndef INCLUDED_REINDEX_HPP
#define INCLUDED_REINDEX_HPP

#include <xtensor/xstrided_view.hpp>
#include <xtensor/xview.hpp>


namespace ij
{
    template<typename E>
    auto i(E&& vector)
    {
        return xt::view(vector, xt::all(), xt::newaxis());
    }
}

namespace ijk
{
    // result(i, j, k) = matrix(i, j)
    template<typename E>
    auto ij(E&& matrix)
    {
        return xt::view(matrix, xt::all(), xt::all(), xt::newaxis());
    }

    // result(i, j, k) = matrix(j, i)
    template<typename E>
    auto ji(E&& matrix)
    {
        return xt::view(xt::transpose(matrix), xt::all(), xt::all(), xt::newaxis());
    }

    // result(i, j, k) = matrix(i, k)
    template<typename E>
    auto ik(E&& matrix)
    {
        return xt::view(matrix, xt::all(), xt::newaxis(), xt::all());
    }

    // result(i, j, k) = matrix(k, j)
    template<typename E>
    auto kj(E&& matrix)
    {
        return xt::view(xt::transpose(matrix), xt::newaxis(), xt::all(), xt::all());
    }
}

#endif
