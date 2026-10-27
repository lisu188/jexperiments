(function () {
    'use strict';

    // Resolve controls through the page structure, never href/data-target strings.
    var dropdownSelector = '.dropdown-toggle[data-toggle="dropdown"]';
    var collapseSelector = '[data-toggle="collapse"]';

    function setDropdown(dropdown, open) {
        dropdown.classList.toggle('open', open);
        dropdown.querySelector(dropdownSelector).setAttribute('aria-expanded', String(open));
    }

    function closeDropdowns(except) {
        document.querySelectorAll('.dropdown.open').forEach(function (dropdown) {
            if (dropdown !== except) {
                setDropdown(dropdown, false);
            }
        });
    }

    function disabled(control) {
        return control.matches(':disabled, .disabled, [aria-disabled="true"]');
    }

    document.querySelectorAll(dropdownSelector + ', ' + collapseSelector).forEach(function (control) {
        control.setAttribute('aria-expanded', 'false');
    });

    document.addEventListener('click', function (event) {
        if (!(event.target instanceof Element)) {
            return;
        }
        var collapse = event.target.closest(collapseSelector);
        if (collapse) {
            event.preventDefault();
            var navbar = collapse.closest('.navbar');
            var panel = navbar && navbar.querySelector('.navbar-collapse');
            if (panel && !disabled(collapse)) {
                var expanded = panel.classList.toggle('in');
                collapse.classList.toggle('collapsed', !expanded);
                collapse.setAttribute('aria-expanded', String(expanded));
            }
            closeDropdowns();
            return;
        }
        var toggle = event.target.closest(dropdownSelector);
        if (toggle) {
            event.preventDefault();
            var dropdown = toggle.closest('.dropdown');
            if (dropdown && !disabled(toggle)) {
                var open = !dropdown.classList.contains('open');
                closeDropdowns(dropdown);
                setDropdown(dropdown, open);
                toggle.focus();
            }
            return;
        }
        closeDropdowns();
    });

    document.addEventListener('keydown', function (event) {
        if (!(event.target instanceof Element)) {
            return;
        }
        var dropdown = event.target.closest('.dropdown');
        var toggle = dropdown && dropdown.querySelector(dropdownSelector);
        if (!toggle || disabled(toggle)) {
            return;
        }
        if (event.key === 'Escape') {
            if (dropdown.classList.contains('open')) {
                event.preventDefault();
                setDropdown(dropdown, false);
                toggle.focus();
            }
            return;
        }
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
            return;
        }
        event.preventDefault();
        closeDropdowns(dropdown);
        setDropdown(dropdown, true);
        var items = Array.from(dropdown.querySelectorAll('.dropdown-menu a')).filter(function (item) {
            return !item.closest('.disabled') && !disabled(item) && item.getClientRects().length > 0;
        });
        if (!items.length) {
            return;
        }
        var index = items.indexOf(document.activeElement);
        index = event.key === 'ArrowDown' ? (index + 1) % items.length :
            (index < 0 ? items.length - 1 : (index - 1 + items.length) % items.length);
        if (items[index]) {
            items[index].focus();
        }
    });

    document.addEventListener('focusin', function (event) {
        if (event.target instanceof Element) {
            closeDropdowns(event.target.closest('.dropdown'));
        }
    });
}());
