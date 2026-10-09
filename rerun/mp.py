"""Plain multiprocessing map for the cache builders: a multiprocessing.Pool created before any other thread exists,
imap with a chunksize. Order preserved.
"""
import multiprocessing as mp


def pmap(func, items, nb_cpus=None, chunksize=None):
    items = list(items)
    if not items:
        return []
    nb_cpus = min(nb_cpus or mp.cpu_count(), len(items))
    if nb_cpus == 1:
        return [func(x) for x in items]
    if chunksize is None:
        chunksize = max(1, min(256, len(items) // (nb_cpus * 8)))
    ctx = mp.get_context('fork')
    with ctx.Pool(nb_cpus) as pool:
        return list(pool.imap(func, items, chunksize=chunksize))
