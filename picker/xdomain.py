"""
xdomain.py  (v17)

Index / offset domain for the picker canvas.

The canvas always works in picker positions (recv_pos: an evenly numbered grid,
one value per recorded trace). The picks are stored in these positions, so
nothing about picking changes. The OFFSET view only changes how the x axis is
drawn: a matplotlib 'function' scale maps each picker position to its true
distance along the line (from the headers or the external topography), so
missing receivers appear as gaps and irregular spacing is drawn to scale.

Around every trace the map is a straight line with the nominal slope (the
wiggles keep their shape); the stretch that makes up a gap happens in the
space between two traces.
"""
import numpy as np


class DomainMap:
    """Monotonic piecewise-linear map picker position -> distance along the line."""

    def __init__(self, recv_pos, recv_dist):
        p = np.asarray(recv_pos, float)
        d = np.asarray(recv_dist, float)
        if p.ndim != 1 or len(p) != len(d) or len(p) < 2:
            raise ValueError('need at least two receivers with positions and distances')
        if not np.all(np.isfinite(d)):
            raise ValueError('some receiver distances are not finite')
        order = np.argsort(p)
        p, d = p[order], d[order]
        if np.any(np.diff(p) <= 0):
            raise ValueError('picker positions repeat')
        dd = np.diff(d)
        if np.all(dd > 0):
            self.sign = 1.0
        elif np.all(dd < 0):
            self.sign = -1.0
        else:
            k = int(np.where(dd * np.sign(np.median(dd)) <= 0)[0][0])
            raise ValueError(f'distances along the line are not in order at traces {order[k] + 1} and '
                             f'{order[k + 1] + 1} ({d[k]:g} m, {d[k + 1]:g} m)')
        self.p, self.d = p, d
        sp_p = float(np.median(np.diff(p)))
        sp_d = float(np.median(np.abs(dd)))
        self.ratio = self.sign * sp_d / sp_p             # distance per position unit
        self.sp_p, self.sp_d = sp_p, sp_d
        # knots: a straight piece of nominal slope around every trace
        gap_l = np.r_[np.inf, np.abs(dd)]
        gap_r = np.r_[np.abs(dd), np.inf]
        w = np.minimum(0.45 * sp_d, 0.45 * np.minimum(gap_l, gap_r))      # half-width in distance
        u = w / abs(self.ratio)                                              # half-width in position
        u = np.minimum(u, 0.45 * np.minimum(np.r_[np.inf, np.diff(p)], np.r_[np.diff(p), np.inf]))
        w = u * abs(self.ratio)
        kp = np.ravel(np.column_stack([p - u, p + u]))
        kd = np.ravel(np.column_stack([d - self.sign * w, d + self.sign * w]))
        self.kp, self.kd = kp, kd
        # inverse needs increasing distances
        if self.sign > 0:
            self.ikd, self.ikp = kd, kp
        else:
            self.ikd, self.ikp = kd[::-1], kp[::-1]
        self.gaps = [(float(p[k]), float(p[k + 1]), float(abs(dd[k])))
                     for k in range(len(dd)) if abs(dd[k]) > 1.5 * sp_d]

    def forward(self, x):
        """Picker position(s) -> distance (m); linear beyond the ends."""
        x = np.asarray(x, float)
        y = np.interp(x, self.kp, self.kd)
        lo, hi = x < self.kp[0], x > self.kp[-1]
        y = np.where(lo, self.kd[0] + (x - self.kp[0]) * self.ratio, y)
        y = np.where(hi, self.kd[-1] + (x - self.kp[-1]) * self.ratio, y)
        return y

    def inverse(self, y):
        """Distance(s) (m) -> picker position."""
        y = np.asarray(y, float)
        x = np.interp(y, self.ikd, self.ikp)
        lo, hi = y < self.ikd[0], y > self.ikd[-1]
        x = np.where(lo, self.ikp[0] + (y - self.ikd[0]) / self.ratio, x)
        x = np.where(hi, self.ikp[-1] + (y - self.ikd[-1]) / self.ratio, x)
        return x

    def describe_gaps(self, limit=6):
        if not self.gaps:
            return 'no gaps'
        s = ', '.join(f'{self.forward(a):.1f}-{self.forward(b):.1f} m' for a, b, _ in self.gaps[:limit])
        return f'{len(self.gaps)} gap(s): {s}' + (' ...' if len(self.gaps) > limit else '')


def make_locator(dmap, nbins=10):
    """Ticks at round distances (in the distance domain) for the offset view."""
    from matplotlib.ticker import Locator, MaxNLocator

    class _DistanceLocator(Locator):
        def __init__(self):
            self._inner = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])

        def __call__(self):
            v0, v1 = self.axis.get_view_interval()
            return self.tick_values(v0, v1)

        def tick_values(self, vmin, vmax):
            d0, d1 = sorted(dmap.forward([vmin, vmax]))
            ticks_d = self._inner.tick_values(d0, d1)
            return dmap.inverse(ticks_d)

    return _DistanceLocator()


def make_formatter(dmap):
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos=None):
        d = float(dmap.forward(v))
        r = round(d, 6)
        return f'{r:g}'
    return FuncFormatter(fmt)


def apply_domain(ax, domain, dmap=None):
    """Set the canvas x axis to 'index' (linear in picker positions) or 'offset'
    (true distances; needs a DomainMap). Keeps the x limits (picker positions)."""
    xl = ax.get_xlim()
    if domain == 'offset' and dmap is not None:
        ax.set_xscale('function', functions=(dmap.forward, dmap.inverse))
        ax.xaxis.set_major_locator(make_locator(dmap))
        ax.xaxis.set_major_formatter(make_formatter(dmap))
        from matplotlib.ticker import NullLocator
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xscale('linear')
        from matplotlib.ticker import AutoLocator, ScalarFormatter, NullLocator
        ax.xaxis.set_major_locator(AutoLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(xl)
