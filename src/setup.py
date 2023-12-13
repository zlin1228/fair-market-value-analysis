from distutils.core import Extension, setup

from Cython.Build import cythonize

ext_modules = [
    ("ema_group_map", ["ema_group_map.pyx"]),
    ("grouped_history", ["grouped_history.pyx"]),
    ("grouped_quotes", ["grouped_quotes.pyx"]),
    ("grouped_ratings", ["grouped_ratings.pyx"])
]

ext_options = {
    "annotate": True,
    "compiler_directives": {"language_level": 3, "profile": True},
}
setup(
    ext_modules=cythonize(
        [
            Extension(
                name,
                dependencies,
                language="c++",
                # NOTE: Cython code is thread-safe
                # extra_compile_args=["-fopenmp"],
                # extra_link_args=["-fopenmp"],
            )
            for (name, dependencies) in ext_modules
        ],
        **ext_options
    )
)