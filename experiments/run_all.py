"""
Run all experiments. Use --quick for a fast test run with fewer simulations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with N_SIMS=5')
    parser.add_argument('--fig', type=str, default='all',
                        help='Which figure to run: 4, 5, 6, 8, or all')
    args = parser.parse_args()

    if args.quick:
        # Patch simulation counts for quick testing
        import experiments.fig4_portfolio_composition as fig4
        import experiments.fig5_out_of_sample as fig5
        import experiments.fig6_performance_comparison as fig6
        import experiments.fig8_radius_vs_N as fig8

        fig4.N_SIMS = 5
        fig4.SAMPLE_SIZES = [30, 300]

        fig5.N_SIMS = 5
        fig5.SAMPLE_SIZES = [30, 300]

        fig6.N_SIMS = 5
        fig6.SAMPLE_SIZES = [30, 100, 300]

        fig8.N_SIMS = 5
        fig8.SAMPLE_SIZES = [30, 100, 300]

    figs_to_run = args.fig

    if figs_to_run in ('all', '4'):
        print("=" * 60)
        print("Figure 4: Portfolio composition vs epsilon")
        print("=" * 60)
        import experiments.fig4_portfolio_composition as fig4
        if args.quick:
            fig4.N_SIMS = 5
            fig4.SAMPLE_SIZES = [30, 300]
        r = fig4.run_experiment()
        fig4.plot_results(r)

    if figs_to_run in ('all', '5'):
        print("=" * 60)
        print("Figure 5: Out-of-sample performance vs epsilon")
        print("=" * 60)
        import experiments.fig5_out_of_sample as fig5
        if args.quick:
            fig5.N_SIMS = 5
            fig5.SAMPLE_SIZES = [30, 300]
        r = fig5.run_experiment()
        fig5.plot_results(r)

    if figs_to_run in ('all', '6'):
        print("=" * 60)
        print("Figure 6: Performance comparison")
        print("=" * 60)
        import experiments.fig6_performance_comparison as fig6
        if args.quick:
            fig6.N_SIMS = 5
            fig6.SAMPLE_SIZES = [30, 100, 300]
        fig6.main()

    if figs_to_run in ('all', '8'):
        print("=" * 60)
        print("Figure 8: Wasserstein radius vs N")
        print("=" * 60)
        import experiments.fig8_radius_vs_N as fig8
        if args.quick:
            fig8.N_SIMS = 5
            fig8.SAMPLE_SIZES = [30, 100, 300]
        r = fig8.run_experiment()
        fig8.plot_results(r)


if __name__ == '__main__':
    main()